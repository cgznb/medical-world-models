"""Roll-to-reference stochastic world model for treatment-contextualized risk.

S0: CT0 plus an explicit treatment/interval/surgery scenario.
S1: the same scenario, updated by legally available CT1.
S2: the same reference transition, optionally updated by real postoperative data.
Without a new observation and with an unchanged scenario/entry, S1 and S2 can
correctly give identical predictions. No artificial stage-specific risk head.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig
from .backbone import ConditionTokens, SpatialTransition, TwoWayFusion, CrossBlock, SetReadout, SurgeryTransition
from .belief import GaussianParams, ObservationPosterior, StochasticInjector, sample_gaussian
from .flow import ConditionalFlowPrior
from .clinical import ClinicalAnchor


def model_from_config(cfg):
    if cfg.architecture == "predictive_ct":
        from .predictive import PredictiveCTWorld
        return PredictiveCTWorld(cfg)
    return TreatmentBeliefWorld(cfg)

class OutcomeReadout(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        h = cfg.hidden
        self.cfg = cfg
        self.fusion = nn.ModuleList([TwoWayFusion(h,cfg.dropout) for _ in range(cfg.readout_blocks)])
        self.pool = SetReadout(h,cfg.readout_slots,cfg.readout_blocks,cfg.dropout)
        self.time = nn.Sequential(nn.Linear(3,h),nn.GELU(),nn.Linear(h,h))
        self.query_blocks = nn.ModuleList([CrossBlock(h,cfg.dropout) for _ in range(cfg.readout_blocks)])
        self.binary_query = nn.Parameter(torch.randn(1,1,h)*.02)
        self.output = nn.Sequential(nn.LayerNorm(h),nn.Linear(h,h),nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(h,cfg.causes))
        self.register_buffer("edges",torch.tensor(cfg.bin_edges,dtype=torch.float))
        nn.init.constant_(self.output[-1].bias,-4. if cfg.endpoint == "survival" else 0.)
    def forward(self,baseline,reference,conditions,entry):
        for layer in self.fusion:
            baseline,reference = layer(baseline,reference)
        memory = self.pool(torch.cat((baseline,reference,reference-baseline,conditions),1))
        if self.cfg.endpoint == "binary":
            q = self.binary_query.expand(len(memory),-1,-1)
        else:
            mid = (self.edges[1:]+self.edges[:-1])/2
            width = self.edges[1:]-self.edges[:-1]
            time = torch.stack((mid[None].expand(len(memory),-1),width[None].expand(len(memory),-1),
                                entry[:,None].expand(-1,len(mid))),-1)/12.
            q = self.time(time)
        for block in self.query_blocks:
            q = block(q,memory)
        result = self.output(q).float()
        return result[:,0,0] if self.cfg.endpoint == "binary" else F.softplus(result)+1e-6

class TreatmentBeliefWorld(nn.Module):
    def __init__(self,cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg.validate()
        h = cfg.hidden
        self.image = nn.Sequential(nn.Linear(cfg.image_dim,h),nn.LayerNorm(h))
        self.condition = ConditionTokens(h)
        coord = torch.stack(torch.meshgrid(*([torch.linspace(-1,1,3)]*3),indexing="ij")).flatten(1).T[None]
        self.register_buffer("coordinates",coord)
        self.position = nn.Linear(3,h,bias=False)
        self.missing_ct = nn.Parameter(torch.zeros(1,27,h))
        self.blocks = nn.ModuleList([SpatialTransition(h,6,cfg.dropout) for _ in range(cfg.world_blocks)])
        self.state_delta = nn.Sequential(nn.LayerNorm(h),nn.Linear(h,h))
        self.prior_pool = SetReadout(h,4,1,cfg.dropout)
        self.prior_params = GaussianParams(h,cfg.latent_dim)
        self.posterior = ObservationPosterior(h,cfg.latent_dim,cfg.update_blocks,cfg.dropout)
        self.injector = StochasticInjector(h,cfg.latent_dim)
        self.decoder = nn.Sequential(nn.LayerNorm(h),nn.Linear(h,2*h),nn.GELU(),nn.Linear(2*h,cfg.image_dim))
        nn.init.normal_(self.decoder[-1].weight,std=.001)
        nn.init.zeros_(self.decoder[-1].bias)
        self.surgery = SurgeryTransition(h,cfg.surgery_blocks,cfg.dropout)
        self.surgery_context = nn.Embedding(4,h)
        self.outcome = OutcomeReadout(cfg)
        self.pcr_pool = SetReadout(h,4,2,cfg.dropout)
        self.pcr_output = nn.Sequential(nn.LayerNorm(h),nn.Linear(h,h),nn.GELU(),nn.Linear(h,1))
        if cfg.clinical_anchor:
            self.recurrence_anchor = ClinicalAnchor()
            self.pcr_anchor = ClinicalAnchor()
            # Begin at the fitted baseline; the world model learns an increment.
            for output in (self.outcome.output[-1], self.pcr_output[-1]):
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)
        self.flow = ConditionalFlowPrior(cfg.latent_dim,h,cfg.flow_blocks) if cfg.prior == "flow" else None
        if cfg.postoperative_dim:
            self.post_projection = nn.Linear(cfg.postoperative_dim,h)
            self.post_blocks = nn.ModuleList([CrossBlock(h,cfg.dropout) for _ in range(cfg.update_blocks)])
            self.post_delta = nn.Linear(h,h)
            nn.init.zeros_(self.post_delta.weight)
            nn.init.zeros_(self.post_delta.bias)
        if cfg.observation_update == "residual":
            self.observation_query_norm = nn.LayerNorm(h)
            self.observation_value_norm = nn.LayerNorm(h)
            self.observation_attention = nn.MultiheadAttention(h,4,dropout=cfg.dropout,batch_first=True)
            self.observation_gate = nn.Parameter(torch.zeros(1,1,h))
        self.register_buffer("tab_mean",torch.zeros(361))
        self.register_buffer("tab_scale",torch.ones(361))
        self.register_buffer("image_mean",torch.zeros(cfg.image_dim))
        self.register_buffer("image_scale",torch.ones(cfg.image_dim))
        self.register_buffer("statistics_fitted",torch.tensor(False))

    @staticmethod
    def raw_condition(batch):
        return torch.cat((batch["clinical"].float(),batch["treatment"].float().flatten(1),
                          torch.log1p(batch["interval_days"][:,None].float()/30)),1)

    @torch.no_grad()
    def fit_statistics(self,training_batch):
        raw = self.raw_condition(training_batch)
        self.tab_mean.copy_(raw.mean(0))
        self.tab_scale.copy_(raw.std(0,unbiased=False).clamp_min(.05))
        available = training_batch["image_valid"][:,0]
        if available.any():
            images = training_batch["ct0"][available].float()
            self.image_mean.copy_(images.mean((0,1)))
            self.image_scale.copy_(images.std((0,1),unbiased=False).clamp_min(.05))
        self.statistics_fitted.fill_(True)

    def encode_image(self,values):
        return self.image((values.float()-self.image_mean)/self.image_scale)+self.position(self.coordinates)

    @torch.no_grad()
    def fit_clinical_anchors(self, training_batch):
        if not self.cfg.clinical_anchor:
            return
        clinical = training_batch["clinical"]
        self.recurrence_anchor.fit(clinical, training_batch["binary"], training_batch["binary_valid"])
        self.pcr_anchor.fit(clinical, training_batch["pcr"], training_batch["pcr_valid"])

    @staticmethod
    def expand_samples(value,m):
        return value[:,None].expand(-1,m,*value.shape[1:]).reshape(-1,*value.shape[1:])

    def reference_transition(self,state,clinical,surgery):
        b,m,l,h = state.shape
        result = self.surgery(state.reshape(b*m,l,h),self.expand_samples(clinical,m),
                              self.expand_samples(surgery,m))
        return result.reshape(b,m,l,h)

    def observation_innovation(self,predicted,observed):
        """A token-set update without assuming anatomical correspondence."""
        query = self.observation_query_norm(predicted)
        memory = self.observation_value_norm(observed)
        delta = self.observation_attention(query,memory,memory,need_weights=False)[0]
        return self.observation_gate.sigmoid()*delta

    def update_postoperative(self,reference,batch):
        if not self.cfg.postoperative_dim or "post" not in batch:
            return reference
        present = batch["post_mask"].any(1)
        if not present.any():
            return reference
        b,m,l,h = reference.shape
        source = reference[present].reshape(-1,l,h)
        # Only rows with >=1 valid token reach attention; no all-masked NaNs.
        post = self.post_projection(batch["post"][present].float())
        post = self.expand_samples(post,m)
        mask = self.expand_samples(~batch["post_mask"][present],m)
        updated = source
        for block in self.post_blocks:
            updated = block(updated,post,key_padding_mask=mask)
        result = reference.clone()
        result[present] = (source+.1*self.post_delta(updated-source)).reshape(-1,m,l,h)
        return result

    def forward(self,batch,samples=4,seed=None,max_stage=2,compute_aux=True,force_gaussian=False):
        if samples < 1 or max_stage not in (0,1,2):
            raise ValueError("Invalid Monte Carlo samples or stage")
        raw = self.raw_condition(batch)
        x = (raw-self.tab_mean)/self.tab_scale
        condition = self.condition(x)
        ct0 = torch.where(batch["image_valid"][:,0,None,None],batch["ct0"],torch.zeros_like(batch["ct0"]))
        initial = self.encode_image(ct0)
        initial = torch.where(batch["image_valid"][:,0,None,None],initial,self.missing_ct)
        tokens = torch.cat((condition,initial),1)
        for block in self.blocks:
            tokens = block(tokens)
        deterministic = initial+self.state_delta(tokens[:,6:])
        pmean,plogvar = self.prior_params(self.prior_pool(deterministic).mean(1))
        generator = None
        if seed is not None:
            generator = torch.Generator(device=ct0.device).manual_seed(seed)
        epsilon = torch.randn((len(ct0),samples,self.cfg.latent_dim),device=ct0.device,generator=generator)
        prior_z = sample_gaussian(pmean,plogvar,epsilon)
        if self.flow is not None and not force_gaussian:
            steps = self.cfg.flow_train_steps if self.training else self.cfg.flow_eval_steps
            prior_z = self.flow.sample(epsilon,deterministic,steps)
        # Compute S0 first. This path does not read CT1 or postoperative tensors.
        generated = self.injector(deterministic,prior_z)
        reference0 = self.reference_transition(generated,x[:,:32],batch["surgery"])
        references = [reference0]
        qmean, qlogvar = pmean, plogvar
        posterior_z = prior_z
        if max_stage >= 1 or compute_aux:
            observed = self.encode_image(batch["ct1"])
            qmean,qlogvar = self.posterior(deterministic,observed)
            posterior_z = sample_gaussian(qmean,qlogvar,epsilon)
            observation_delta = None
            if self.cfg.observation_update == "residual" and max_stage >= 1:
                observation_delta = self.observation_innovation(deterministic,observed)
            for stage in range(1,max_stage+1):
                legal = batch["image_valid"][:,1] & (batch["ct1_available_stage"] <= stage)
                z = torch.where(legal[:,None,None],posterior_z,prior_z)
                state = self.injector(deterministic,z)
                if observation_delta is not None:
                    state = state+torch.where(legal[:,None,None,None],observation_delta[:,None],0.)
                reference = self.reference_transition(state,x[:,:32],batch["surgery"])
                if stage == 2:
                    reference = self.update_postoperative(reference,batch)
                references.append(reference)
        pcr_condition = self.expand_samples(condition,samples)  # preoperative head cannot read surgery
        condition = torch.cat((condition,self.surgery_context(batch["surgery"])[:,None]),1)
        entry = batch.get("entry",torch.zeros((len(ct0),3),device=ct0.device))
        flat_initial = self.expand_samples(initial,samples)
        flat_condition = self.expand_samples(condition,samples)
        predictions = []
        for stage,reference in enumerate(references):
            output = self.outcome(flat_initial,reference.flatten(0,1),flat_condition,
                                  self.expand_samples(entry[:,stage],samples))
            predictions.append(output.reshape(len(ct0),samples,*output.shape[1:]))
        result = {"predictions": torch.stack(predictions,1), "pmean": pmean,"plogvar": plogvar,
                  "qmean": qmean,"qlogvar": qlogvar}
        if self.cfg.clinical_anchor:
            result["predictions"] = (self.recurrence_anchor(batch["clinical"])[:,None,None]
                                     + self.cfg.residual_scale*result["predictions"])
        if compute_aux:
            qstate = self.injector(deterministic,posterior_z[:,:1])[:,0]
            result["features"] = ct0+self.decoder(qstate)*self.image_scale
            result["prior_features"] = ct0+self.decoder(generated[:,0])*self.image_scale
            pcr_memory = torch.cat((flat_initial,generated.flatten(0,1),pcr_condition),1)
            result["pcr_logits"] = self.pcr_output(self.pcr_pool(pcr_memory).mean(1)).reshape(len(ct0),samples)
            if self.cfg.clinical_anchor:
                result["pcr_logits"] = (self.pcr_anchor(batch["clinical"])[:,None]
                                        + self.cfg.residual_scale*result["pcr_logits"])
            if self.flow is not None:
                result["flow_loss"] = self.flow.loss(posterior_z[:,0],deterministic,generator)
        return result
