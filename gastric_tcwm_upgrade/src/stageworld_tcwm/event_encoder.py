"""Seven modality fields, explicit metadata, and causal event history."""
import torch
from torch import nn
from .backbone import CrossBlock


class EventEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        self.time_basis = cfg.time_basis
        self.modality_embedding = nn.Embedding(7, h)
        self.status_embedding = nn.Embedding(3, h)
        self.phase_embedding = nn.Embedding(5, h)
        self.operation_embedding = nn.Embedding(7, h)
        self.role_embedding = nn.Embedding(4, h)
        self.time_projection = nn.Sequential(nn.Linear(6, h), nn.SiLU(), nn.Linear(h, h))
        self.ordinal_position = nn.Sequential(nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h))
        self.unknown_time = nn.Parameter(torch.randn(1, h) * .02)
        self.field_blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            h, 4, 4*h, cfg.dropout, "gelu", batch_first=True, norm_first=True)
            for _ in range(cfg.field_blocks)])
        self.pool_queries = nn.Parameter(torch.randn(1, 2, h) * .02)
        self.pool = CrossBlock(h, cfg.dropout)
        self.history_blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            h, 4, 4*h, cfg.dropout, "gelu", batch_first=True, norm_first=True)
            for _ in range(cfg.history_blocks)])
        self.register_buffer("schema_enabled", torch.tensor(cfg.schema_enabled, dtype=torch.bool))

    def encode_local(self, batch):
        values, known = batch["modality_value"], batch["modality_known"].bool()
        mask = batch["event_mask"].bool()
        if (batch["modality_applicable"].bool() & ~self.schema_enabled & mask[..., None]).any():
            raise ValueError("An unsupported modality cannot be applicable to a real event")
        applicable = batch["modality_applicable"].bool() & self.schema_enabled
        if values.ndim != 3 or values.shape[-1] != 7:
            raise ValueError("Event fields must have shape [B,L,7]")
        if ((values != 0) & (values != 1) & known & applicable).any():
            raise ValueError("Known modality values must be binary")
        status = torch.where(known, values.long() + 1, torch.zeros_like(values, dtype=torch.long))
        # Inapplicable values are structurally masked and cannot influence embeddings.
        status = torch.where(applicable, status, torch.zeros_like(status))
        tokens = self.modality_embedding.weight[None, None] + self.status_embedding(status)
        time_features = batch["time_features"].float()
        if time_features.shape != (*mask.shape, 6) or not torch.isfinite(time_features[mask]).all():
            raise ValueError("Each real event requires six finite, legally available time features")
        if self.time_basis == "ordinal_stage":
            # No day-scaled features enter the explicit stage-order path.
            time = self.unknown_time + self.ordinal_position(batch["event_order"].float().unsqueeze(-1))
        else:
            time = self.time_projection(time_features)
        metadata = torch.stack((self.phase_embedding(batch["phase"].long()),
                                self.operation_embedding(batch["operation"].long()),
                                self.role_embedding(batch["role"].long()), time), dim=2)
        tokens = torch.cat((tokens, metadata), dim=2)
        padding = torch.cat((~applicable, torch.zeros((*mask.shape, 4), dtype=torch.bool, device=mask.device)), -1)
        b, length = mask.shape
        result = tokens.new_zeros((b*length, 2, tokens.shape[-1]))
        flat_mask = mask.flatten()
        if flat_mask.any():
            encoded = tokens.reshape(b*length, 11, -1)[flat_mask]
            selected_padding = padding.reshape(b*length, 11)[flat_mask]
            for block in self.field_blocks:
                encoded = block(encoded, src_key_padding_mask=selected_padding)
                encoded = encoded.masked_fill(selected_padding[..., None], 0.)
            result[flat_mask] = self.pool(self.pool_queries.expand(len(encoded), -1, -1),
                                          encoded, key_padding_mask=selected_padding)
        return result.reshape(b, length, 2, tokens.shape[-1])

    def encode_history(self, vectors, event_mask):
        b, length, h = vectors.shape
        output = torch.zeros_like(vectors)
        if not length or not event_mask.any():
            return output
        # Packing removes PAD and duplicate records entirely, including from history positions.
        counts = event_mask.sum(1)
        selected = (counts > 0).nonzero(as_tuple=True)[0]
        width = int(counts.max().item())
        packed = vectors.new_zeros((len(selected), width, h))
        padding = torch.ones((len(selected), width), dtype=torch.bool, device=vectors.device)
        for row, original in enumerate(selected):
            count = int(counts[original].item())
            packed[row, :count] = vectors[original, event_mask[original]]
            padding[row, :count] = False
        causal = torch.triu(torch.ones(width, width, dtype=torch.bool, device=vectors.device), diagonal=1)
        for block in self.history_blocks:
            packed = block(packed, src_mask=causal, src_key_padding_mask=padding)
            packed = packed.masked_fill(padding[..., None], 0.)
        for row, original in enumerate(selected):
            output[original, event_mask[original]] = packed[row, :counts[original]]
        return output

    def forward(self, batch):
        local = self.encode_local(batch)
        history = self.encode_history(local.mean(2), batch["event_mask"].bool())
        return local, history
