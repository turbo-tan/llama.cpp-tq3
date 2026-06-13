# TQ Policy Plan for 9B and 122B

Date: 2026-04-02

Goal:
- stop hand-writing one-off override files
- use a structured policy generator for the next dense and MoE lines

Generator:
- [generate_tq_policy.py](/home/awee/code/tan_llama/scripts/generate_tq_policy.py)

Current starter outputs:
- [9B_TQ_POLICY_balanced.txt](/home/awee/code/tan_llama/docs/turboquant/generated/9B_TQ_POLICY_balanced.txt)
- [9B_TQ_POLICY_compact.txt](/home/awee/code/tan_llama/docs/turboquant/generated/9B_TQ_POLICY_compact.txt)
- [122B_TQ_POLICY_balanced.txt](/home/awee/code/tan_llama/docs/turboquant/generated/122B_TQ_POLICY_balanced.txt)
- [122B_TQ_POLICY_compact.txt](/home/awee/code/tan_llama/docs/turboquant/generated/122B_TQ_POLICY_compact.txt)

Principle:
- edge layers get higher precision
- near-edge layers get medium precision
- middle layers get the aggressive compression
- dense and MoE use different tensor families

9B dense direction:
- start from `balanced`
- use it as the first structured replacement for ad hoc `edge` and `topN ffn` overrides
- if quality is strong, push toward `compact`

122B MoE direction:
- start from `balanced`
- keep shared experts higher precision
- compress routed experts harder in the middle
- only move to `compact` after a short quality gate

Practical next step:
1. generate the policy
2. quantize one balanced model
3. run short quality gate
4. only then try compact / mini

Important:
- this is policy infrastructure, not a final claim
- APEX is the pattern to learn from, not a config to copy blindly
