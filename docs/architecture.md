# Next-GenAI Architecture

Reference: Ali, B., & Chen, G. (2024). *Next-generation AI for advanced threat
detection and security enhancement in DNS over HTTPS*. JNCA, Elsevier.

## 1. High-Level Flow (Fig. 1 of paper)

DoH Traffic (TLS 1.3 + HTTP/2/3)
│
▼
┌──────────────────────────────────┐
│ Feature Engineering (Sec. 3.1) │
│ • DNS query patterns │
│ • TLS metadata │
│ • DoH HTTP headers │
│ • Temporal patterns │
│ → PCA(64) → t-SNE(2) [visual] │
└──────────────┬───────────────────┘
▼
┌──────────────────────────────────────────────┐
│ AI Detection Engine │
│ ┌─────────────┐ ┌──────────────┐ ┌────────┐ │
│ │ CapsNet │ │ GraphTF │ │ CSSL │ │
│ │ Eq. 4–8 │ │ Eq. 9–14 │ │ Eq. 15 │ │
│ │ 99.1% acc │ │ 98.8% recall │ │0.992 AUC│ │
│ └──────┬──────┘ └──────┬───────┘ └───┬────┘ │
│ └────────┬──────┴─────────────┘ │
│ ▼ │
│ Weighted ensemble (0.45/0.40/0.15) │
└──────────────────┬───────────────────────────┘
▼
┌──────────────────────────────────────┐
│ Explainability (Sec. 3.6) │
│ SHAP (Eq. 16) + LIME │
└──────────────────┬───────────────────┘


## 2. Module Map

| Paper Sec. | Implementation |
|-----------|----------------|
| 3.1 | `src/data_acquisition/preprocessing.py` |
| 3.2 | `src/threat_modeling/attack_simulator.py` |
| 3.3 | `src/models/capsule_network.py` |
| 3.4 | `src/models/graph_transformer.py` |
| 3.5 | `src/models/cssl.py` |
| 3.6 | `src/explainability/shap_explainer.py` |
| 1, 2.1 | `src/streaming/flink_pipeline.py` |
| 2.3, 3.7 | `src/security/tee_manager.py` |

## 3. Equation → Code Map

| Eq. | Formula | Implementation |
|-----|---------|----------------|
| 1 | `M_i = {X : Ψ(x,t)=0}` | `capsule_network.CapsNet` input manifold |
| 2 | `t: A ↪ M_i` | `attack_simulator.AttackSimulator` |
| 3 | `∇Ψ·δΨ > ε` | `AttackSimulator.satisfies_evasion` |
| 4 | Ricci flow routing | `RicciFlowRouting._ricci_flow_step` |
| 5 | `α_ij = softmax(angle(...))` | `RicciFlowRouting._geometric_consistency` |
| 6 | `c_ij = exp(b_ij)/Σ...` | `RicciFlowRouting.forward` |
| 7 | Margin loss | `MarginLoss.forward` |
| 8 | Squash | `capsule_network.squash` |
| 9 | Scaled dot-product attention | `GraphAttention.forward` |
| 10 | Frobenius anomaly score | `frobenius_anomaly_score` |
| 11 | Graph-attention bias | `GraphAttention.forward` (W_a) |
| 12 | Node update | `GraphAttention.forward` (activation) |
| 13 | Sheaf morphism | `SheafLaplacian.forward` |
| 14 | Covariant derivative | `SheafLaplacian.forward` (reg term) |
| 15 | NT-Xent loss | `cssl.nt_xent_loss` |
| 16 | Shapley value | `shap_explainer.shapley_value_exact` |
