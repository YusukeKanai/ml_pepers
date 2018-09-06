# Self-Imitation Learning

[arxiv](https://arxiv.org/pdf/1806.05635v1.pdf)

[github](https://github.com/junhyukoh/self-imitation-learning)

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

+++

## Exploration

- 最近では好奇心や不確定性を探索に扱うことが多い
- 探索のため利用の役割の研究は従来からされているが、「学習したものを利用する」が主であり、今回の論文では「学習はしていないが経験した」を対象としている

+++

## Episodic control

- 過去の良かった経験を活かす極端な方法である
- 実用するにはstep毎に関連するstateを検索するため動作が遅いという問題や、
non-parametric policyなので汎化性能がよくないと思われる。

+++

## Experience replay

- TD errorに基づいて過去の経験に優先順位をつける手法:Priority experience replayが有効な手段として過去に提案されている
- - Self-Imitation Learningでも活用している
- Priority experience replayはSarsaやQ学習のような価値反復に基づく学習には有効であることは示されたが、
actor-criticのような方策勾配に基づく学習への適応は難しかった

+++

## Experience replay for actor-critic

- actor-criticも過去の経験を利用してはいるが、その多くは方策オフ型のactor-criticである.
- 過去のpolicyと現行のpolicyが異なるということはよくあり(方策の使い回しが行われない)、そのため上手く学習できないことがある

+++

## Connection between policy gradient and Q-learning

- 最近の研究で方策勾配とentropy正則化付きQ学習が密な関係であることが知られている
- Q学習そのものを使った研究は既存にはあるが、本論文ではlower bound Q学習を利用するところが新しい

+++

## Learning from imperfect demonstrations

- 強化学習タスクでは限られたデータで学習するということが往々にして扱われる
- 自身の経験を学習データとして扱う.
- 似たような研究は他にもあるが理論的背景が十分出ないことがある.本論文では理論的背景も正当化している

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

## Prioritized Replay

効率的に学習が進むように経験再生の順位つけを行われている
具体的には経験の選択確率は`\((R-V_{\theta }(s))_+\)`に比例する。

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

---

## 実験

- Key-Door-Treasure Domain
- Hard Exploration Atari Games
- Overall Performance on Atari Games
- Effect of Lower Bound Soft Q-Learning
- Performance on MuJoCo

---

## 実装

- 3層のDQN
- 入力4frame (SIL学習は4回繰返す)
- ゲームの終了をエピソードの終端にした
- MuJoCoではMLPを2層64unitとした
- MuJoCoではSIL学習を10回繰り返した

---

## Key-Door-Treasure Domain

- 探索ボーナスを追加 `\(r_{\it exp} = \beta / \sqrt{N(s)}\)`
-  SILによって学習が早く進んでいる.(良い経験が生かされている)

![Key-Door-Treasure](SelfImitationLearning/assets/Key-Door-Treasure.png)

+++

## Apple-Key-Door-Treasure

- 50ステップ以内により多くのりんごを集めてドアを開けるゲーム
- 探索ボーナスがあることで宝を取得できるようになっている
- SILは短期的な探索に探索ボーナスは長期的な探索に使っており、相補的となっている

---

## Hard Exploration Atari Games

- 探索が困難なAtariのゲームにおいてSILがあった方が有利に働きやすかった

![compare_result_graph](SelfImitationLearning/assets/compare_result_graph.png)

+++

- 他のアルゴリズムと比較しても良い成績となりやすかったが「Venture」のみ成績が悪かった.
良い経験を一度も得られなかったからだと考えられる

![compare_result](SelfImitationLearning/assets/compare_result.png)

---

## Overall Performance on Atari Games

- 6/7 のAtariのゲームでSILが有効に働いた

+++

![sil_effect](SelfImitationLearning/assets/sil_effect.png)

+++

- A2Cのみの方がいい結果のものもある.ゲーム初期の経験が後半に活かせないゲームもあり、これがネックになっていそう。
- - SILの更新回数を減らすか目的関数のSIL項を小さくするパラメタを導入することで解決できると考えられる.

![sil_outperforms](SelfImitationLearning/assets/sil_outperforms.png)

---

##  Effect of Lower Bound Soft Q-Learning

- 方策オフ型actor-criticを使えばSILじゃなくて十分性能が出るのか確認
- - ACPERで試したところ十分な性能が出なかった

---

##  Performance on MuJoCo

- 連続操作においてSILが有効か確認した。
- - A2Cの代わりにPPOを使用(ただし理論的に結び付けられる強い根拠はない)
- - 改善できるタスクは限られている。ゲーム中に良い経験を得る機会が少ないから

![mujoco_task](SelfImitationLearning/assets/ppo_sil.png)
