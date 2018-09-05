# Self-Imitation Learning

[arxiv](https://arxiv.org/pdf/1806.05635v1.pdf)

---

## この論文の概要

- A simple off-policy actor-critic algorithm
- 過去の経験を再生成しようとすることで、さらに深い探索を行う

---

## 探索と利用のトレードオフ

![トレードオフ](SelfImitationLearning/assets/exploration_exploition_trade-off.png)

_[Reinforce Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) p.3より_

---

## 関連研究

---

## Self-Imitation Learning

actor-critic法によって過去の良い経験を模倣しようとする

`\(s_t\)`: a state at time-step \(t\) <br>
`\(a_t\)`: an action at time-step \(t\) <br>
`\(R_t = \Sigma^{\infty}_{k=t}\gamma^{k-t}r_{k}\)`: discounted sum of rewards <br>
`\(\gamma\)`: discount factor

`\({\cal D} = \{(s_t, a_t, R_t)\}\)`: Replay Buffer

---

## SILの目的関数

`\[
\begin{aligned}
{\cal L^{\it sil}} &= \mathbb{E}_{a, a, R \in {\cal D}}[{\cal L^{\it sil}_{policy}} + \beta^{\it sil}{\cal L^{\it sil}_{value}}] \\
{\cal L^{\it sil}_{policy}} &= -\log \pi_{\theta}(a|s)(R-V_{\theta}(s))_{+} \\
{\cal L^{\it sil}_{value}} &= \frac{1}{2}\|(R-V_{\theta}(s))\|^2 \\
\end{aligned}
\]`
where <br>
`\((\cdot)_{+} = \max(\cdot, 0)\)` <br>
`\(\pi_{\theta}\): policy, \(V_{\theta}(s)\)`: value function <br>
`\(\beta^{\it sil} \in \mathbb{R}^+\)`

+++

## `\({\cal L^{\it sil}_{policy}}\)` の解釈

1. 行動価値関数`\(Q(s,a)\)` をモンテカルロ収益`\(R\)`に近似し、状態価値関数`\(V(s)\)`をベースラインとした方策勾配とみなせる
2. サンプルウェイトが`\(R-V_{\theta}(s)\)`に比例したクロスエントロピーロスとみなせる

`\(R > V_{\theta}(s)\)` とエージェントが推定すれば過去のその行動は採用されやすいよう学習される。一方で`\(R < V_{\theta}(s)\)` であれば何もされない(学習が進まない)

+++

## `\({\cal L^{\it sil}_{value}}\)` の解釈

推定値`\(V_{\theta}(s)\)` をRへ近づける

---

## SILの理論的裏付け

### Claim

_The self-imitation learning objective can be viewed as an implementation of
lower-bound-soft-Q-learning under entropy-regularlized RL framework__

+++

## Entropy-Regularized Reinforcement Learning

`\[
\pi ^{*} = \text{argmax} _{\pi} \mathbb{E}_{\pi}[\Sigma ^{\infty}_{t=0}\gamma ^{t}(r_t +\alpha {\cal H}^{\pi}_{t})]
\]`

where <br>
`\({\cal H}^{\pi}_{t} = -\log {\pi}(a_t|s_t)\)`: entropy of the policy `\(\pi\)` <br>
`\(\alpha \geq 0\)` : the weight of entropy bonus
