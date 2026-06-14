# Formal Neuroscience Framework for ACC Conflict Detection

## 1. Core Hypothesis

**The Anterior Cingulate Cortex (ACC) detects conflict between predicted and observed neural states and recruits cognitive control to resolve it.** We translate this into a language model as follows:

> A large language model hallucinates or becomes overconfident when its generative predictions become decoupled from the evidence encoded in its internal representations. The ACC-inspired detector monitors the *conflict* between (a) the model's top-down token prediction and (b) the bottom-up hidden-state evidence, and intervenes when conflict exceeds a learned threshold.

## 2. Biological Inspiration

### 2.1 Predictive Coding (Friston, 2005; Rao & Ballard, 1999)

Cortex is hypothesized to minimize prediction error:

```
Prediction error = Observed activity − Predicted activity
```

In our architecture:
- **Observed activity** = hidden state at layer *l*, token *t*.
- **Predicted activity** = hidden state projected forward from layer *l−1*.
- **Prediction error** = residual of a learned projection.

This maps directly onto the predictive coding objective:

```
ε_l(t) = h_l(t) − f_l(h_{l−1}(t))
```

where `f_l` is a learned linear projection (the cortical prediction).

### 2.2 Conflict Monitoring Theory (Botvinick et al., 2001; Yeung et al., 2004)

The ACC signals the co-activation of incompatible responses. In language generation, incompatible responses arise when:
1. The output distribution assigns high probability to a token.
2. The internal hidden-state evidence (lower layers / early tokens) contradicts that token.

We quantify conflict as the *divergence* between the predicted next-token distribution and the actual next-token distribution implied by the current hidden state.

### 2.3 Hierarchical Processing (Kiebel et al., 2008)

Predictive coding is hierarchical: lower levels predict fast sensory details; higher levels predict slower semantic structure. We mimic this by:
- Computing prediction errors at multiple layer pairs.
- Aggregating them with a learned temporal integrator (leaky integrator = persistent activity).

## 3. Mathematical Formalization

### 3.1 Hidden-State Prediction Error

For a model with layers `l = 1 … L`, hidden states `h_l(t) ∈ R^d`, we define a learned projection `W_l ∈ R^{d×d}` and bias `b_l ∈ R^d`:

```
ĥ_l(t) = W_l · h_{l−1}(t) + b_l
ε_l(t) = h_l(t) − ĥ_l(t)
```

The projection is trained to minimize `||ε_l(t)||²` on the model's own pretraining distribution, analogous to learning the cortical prediction function.

### 3.2 Next-Token Prediction Error

The language model emits logits `z(t) ∈ R^V`. We project the hidden state at layer *l* to a distribution `p_l(t)`:

```
p_l(t) = softmax(O_l · h_l(t) + c_l)
```

The prediction error at the output level is the KL divergence between the final output distribution `p_L(t)` and the intermediate distribution `p_l(t)`:

```
D_l(t) = KL(p_L(t) || p_l(t))
```

High `D_l(t)` means the intermediate layer disagrees with the final output.

### 3.3 Conflict Score

The detector combines hidden-state prediction error and output disagreement into a scalar conflict score:

```
s(t) = MLP([ ε_l(t); D_l(t); Δε_l(t) ])
```

where `Δε_l(t)` is the temporal derivative (leaky integrator state) capturing persistent conflict.

### 3.4 Temporal Integration

Biological ACC shows sustained activity after conflict. We implement a leaky integrator:

```
r(t) = α · r(t−1) + (1 − α) · s(t)
```

with `α ∈ [0,1]` learned or fixed. The integrated signal `r(t)` is used for intervention decisions.

## 4. Mapping to Model Components

| Neuroscience Concept | Implementation | File |
|----------------------|----------------|------|
| Top-down prediction | Linear projection of hidden state | `src/acc_layer.py` |
| Prediction error | Hidden-state residual | `src/acc_layer.py` |
| Output disagreement | KL between final and intermediate logits | `src/acc_conflict_detector.py` |
| Persistent ACC activity | Leaky integrator | `src/acc_conflict_detector.py` |
| Cognitive control intervention | Logit shifting / regeneration | `src/acc_intervention.py` |
| Response conflict threshold | Learned classifier threshold | `src/acc_conflict_detector.py` |

## 5. Why This Is Not Just "Attention" or "Entropy"

| Method | What it measures | Limitation |
|--------|------------------|------------|
| Entropy | Output uncertainty | Cannot detect *confident* hallucinations |
| Perplexity | Sequence likelihood | Punishes complexity, rewards overconfidence |
| DoLa | Contrastive logit difference | Only compares two layer outputs |
| SAPLMA | Hidden-state classifier | No temporal dynamics, no prediction error |
| **ACC (ours)** | Hierarchical prediction-error conflict + temporal integration | Detects decoupling between prediction and evidence |

## 6. Testable Predictions

Our framework makes concrete empirical predictions:

1. **Prediction-error conflict correlates with hallucination.** Hallucinated tokens should have higher `ε_l(t)` and `D_l(t)` than supported tokens.
2. **Temporal integration improves detection.** A model with leaky integration should outperform a frame-by-frame detector on long-form generation.
3. **Layer-pair selection matters.** Early-to-mid layers (where semantic and syntactic processing interact) should be most predictive.
4. **Intervention should reduce hallucination.** Logit-shifting toward uncertainty tokens should increase refusal/hedging on hallucination prompts.
5. **Ablation of prediction error hurts performance.** Removing the hidden-state residual should degrade detection more than removing the KL term.

## 7. Limitations and Caveats

- The mapping is *analogical*, not literal. We do not claim the model implements biological neurons.
- The learned projection `W_l` is task-specific; biological prediction is likely more structured.
- We do not model neuromodulatory effects (dopamine, norepinephrine) that modulate ACC gain.

## 8. Suggested Paper Text (Methods)

> We frame hallucination detection as **conflict monitoring under predictive coding**. At each generation step, the model's hidden states encode both a top-down prediction of the next token and bottom-up evidence from previous context. Our detector computes the prediction error between adjacent-layer hidden states and the divergence between intermediate and final output distributions. A leaky temporal integrator accumulates conflict over time, mimicking the sustained activity observed in the anterior cingulate cortex during response conflict. When the integrated conflict score exceeds a learned threshold, an intervention shifts the output distribution toward uncertainty tokens, analogous to cognitive control recruiting cautious responding.

## 9. References

- Botvinick, M. M., Braver, T. S., Barch, D. M., Carter, C. S., & Cohen, J. D. (2001). Conflict monitoring and cognitive control. *Psychological Review*, 108(3), 624.
- Friston, K. (2005). A free energy principle for the brain. *Journal of Physiology-Paris*, 100(1-3), 70-87.
- Kiebel, S. J., Daunizeau, J., & Friston, K. J. (2008). A hierarchy of time-scales and the brain. *PLoS Computational Biology*, 4(11), e1000209.
- Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79-87.
- Yeung, N., Botvinick, M. M., & Cohen, J. D. (2004). The neural basis of error detection: conflict monitoring and the error-related negativity. *Psychological Review*, 111(4), 931.
