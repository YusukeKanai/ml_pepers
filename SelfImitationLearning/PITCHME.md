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
lower-bound-soft-Q-learning under entropy-regularlized RL framework_

+++

## Entropy-Regularized Reinforcement Learning

最適な方策`\(\pi ^{*} \)`を求めたい

`\[
\pi ^{*} = \text{argmax} _{\pi} \mathbb{E}_{\pi}[\Sigma ^{\infty}_{t=0}\gamma ^{t}(r_t +\alpha {\cal H}^{\pi}_{t})]
\]`

where <br>
`\({\cal H}^{\pi}_{t} = -\log {\pi}(a_t|s_t)\)`: entropy of the policy `\(\pi\)` <br>
`\(\alpha \geq 0\)` : the weight of entropy bonus

Entropyによって __ある状態での取りうる行動が分散された方がいい__ を達成する。(探索する)

+++

## optimal soft Q-function / optimal soft value function

`\[
\begin{aligned}
Q^{*}(s_t, a_t) &= \mathbb{E}_{\pi ^{*}}[r_t + \Sigma ^{\infty}_{k=t+1} \gamma ^{k-t}(r_k + \alpha {\cal H}^{\pi ^{*}}_k)] \\ 
V ^{*} (s_t) &= \alpha \log {\Sigma _{\alpha }\exp {(Q ^{*}(s_t, a)/{\alpha })}}
\end{aligned}
\]`

この定義の元で以下の関係式が導出されることが知られている。

`\[
\pi ^{*}(a|s) = \exp ((Q ^{*}(s, a)-V ^{*}(s))/\alpha)
\]`

+++

## Lower bound soft Q-learning

最適方策は他の方策よりも価値の高いため、任意の方策`\(\mu \)`について以下が成り立つ

`\[
\begin{aligned}
Q^{*}(s_t, a_t) &= \mathbb{E}_{\pi ^{*}}[r_t + \Sigma ^{\infty}_{k=t+1} \gamma ^{k-t}(r_k + \alpha {\cal H}^{\pi ^{*}}_k)] \\ 
&\geq \mathbb{E}_{\mu }[r_t + \Sigma ^{\infty}_{k=t+1} \gamma ^{k-t}(r_k + \alpha {\cal H}^{\mu }_k)]
\end{aligned}
\]`

+++

方策`\(\mu \)`に対して以下の目的関数を用意する

`\[
{\cal L}^{lb} = \mathbb{E}_{s,a,R \sim \mu }[\frac{1}{2} \|(R-Q_{\theta }(s,a))_+\|^2]
\]`

これはある行動`\(a \)`をとった時の報酬の期待値が、実際の報酬より小さければその行動は修正されるべきであると解釈できる。

+++

以下、先ほどの最適解を求めようとしよう、すると以下が導かれる

`\[
\begin{aligned}
\nabla {\cal L}^{\it lb} = \mathbb{E}[\alpha \nabla _\theta {\cal L}^{\it lb}_{policy} + \nabla _\theta {\cal L}^{\it lb}_{value}]
\end{aligned}
\]`

ここで

`\[
\begin{aligned}
{\cal L^{\it lb}_{policy}} &= -\log \pi_{\theta}(a|s)(\hat{R}-V_{\theta}(s))_{+} \\
{\cal L^{\it lb}_{value}} &= \frac{1}{2}\|(\hat{R}-V_{\theta}(s))\|^2 \\ 
\hat{R} &= R-\alpha \log \pi _\theta (a|s)
\end{aligned}
\]`

+++

さらに

`\(\alpha \rightarrow 0\)` とすると`\({\cal L^{\it lb}_{policy}} = {\cal L^{\it sil}_{policy}}, {\cal L^{\it lb}_{value}} – {\cal L^{\it sil}_{value}}\)`が得られる。

SILアルゴリズムはlower bound of the optimal Q-valueを求めることに一致することを意味する。

---

## A2Cとの結合

A2Cの目的関数
`\[
\begin{aligned}
{\cal L^{\it a2c}} &= \mathbb{E}_{s, a \sim \pi _\theta }[{\cal L^{\it a2c}_{policy}} + \beta ^{\it a2c}{\cal L^{\it a2c}_{value}}] \\
{\cal L^{\it a2c}_{policy}} &= -\log \pi_{\theta}(a_t|s_t)(V^{n}_{t}-V_{\theta}(s_t))-\alpha {\cal H}^{\pi_{\theta}}_{t} \\
{\cal L^{\it a2c}_{value}} &= \frac{1}{2}\|(V_{\theta}(s)-V^{n}_{t})\|^2 \\
\end{aligned}
\]`
where
`\[
V^n_t = \Sigma ^{n-1}_{d=0}\gamma ^{d}r_{t+d} + \gamma ^{n}V_{\theta }(s_{t+n})]
\]`
+++

## A2CとSILの関係

- A2Cは方策オン型の強化学習であり、より良い方策を探索する(explore)
- SILは方策オフ型の強化学習であり、過去の経験から適した方策を利用する(exploit)
- 共にエントロピー正則化強化学習の元で最適ソフトQ関数を求めようとしている <br>
- 相補的に最適方策を求めようとしている

---

## アルゴリズム

![擬似コード](SelfImitationLearning/assets/psuedo_code.png)
