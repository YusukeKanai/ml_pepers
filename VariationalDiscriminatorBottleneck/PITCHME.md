# Variational Discriminator Bottleneck

## Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow

---

- [arxiv](https://arxiv.org/pdf/1810.00821v1.pdf)
- [openreview](https://openreview.net/forum?id=HyxPx3R9tm)
- [github](https://github.com/akanimax/Variational_Discriminator_Bottleneck)

---

## Agenda

- 敵対的学習の紹介
- 先人の知恵: これまでのGANの学習を安定にさせた努力
- 未だ残る課題: 何がたりなのか
- 新しい仕組みの発明: 情報を与えないことが大事
- 新しい仕組みを試す: 本当に上手くいくのか
- まとめ
- Appendix:

---

## 敵対的学習の紹介

- 高次元で複雑な構想をしたデータをモデル化してくれる
- Generatorで上手くfakeデータを生成し, Discriminatorでfakeデータとrealデータを識別する学習をするゲーム理論的な戦略をとっている
- 逆強化学習や模倣学習にも応用されている

+++

 - GeneratorとDiscriminatorのバランスが難しい
   - 厳しいDiscriminatorは不要な情報まで学習しようとする
   - ゆるいDiscriminatorはGeneratorの学習能力を弱いものにする

\\

__不要な情報をどう取り除くかがポイント__

---

この研究では __対象物をDiscriminatorに渡す際に制限を課す.__ この制限は __変数近似をInformation Bottleneck法に適応すること__ で実現する

---

# Information Bottleneck Method

[The Information Bottleneck Method](https://www.cs.huji.ac.il/labs/learning/Papers/allerton.pdf)
(99' Tishby et al.)

[DEEP VARIATIONAL INFORMATION BOTTLENECK](https://arxiv.org/pdf/1612.00410v5.pdf)
(17' A.A. Alemi et al)

+++

## 説明 (by 朱鷺の森 wiki)

__[Information Bottleneck Method (情報ボトルネック)]__

_クラスタリングされる離散確率変数Xと,Xに関連した情報Yを考える.例えばXが単語集合で,Yは文書クラスなど.ここでX中の要素を，それらと関連情報Yの類似するように分割する.例えば,文書クラスを関連情報としたとき,同じクラスに現れやすい単語クラスタを獲得できたりする._

_形式的には次のような問題：XをX̃に分割する.この分割は,確率的写像 Pr[X̃|X]で表し,次式を最小化するように定める._

`\[
{\cal L}(\Pr [\tilde X | X]) = \text{I}(X;\tilde X)-\beta \text{I}(\tilde X|Y)
\]`

(引用:[朱鷺の森](http://ibisforest.org/index.php?%E6%83%85%E5%A0%B1%E3%83%9C%E3%83%88%E3%83%AB%E3%83%8D%E3%83%83%E3%82%AF))

+++

![IB](VariationalDiscriminatorBottleneck/assets/IB.png)

`\(\min {\cal L}(\Pr [\tilde X | X])\)` : 出来るだけ入力情報を圧縮すると同時に圧縮された情報が出力情報にとって意味あるものにする

---

## Variational Information Bottleneck (A.A.Alemi et al)

- Information Bottleneck法を教師あり学習に応用した

__目的函数__

`\[
\min_{q, E} \{
  \mathbb{E}_{\mathbf{x,y} \sim p(\mathbf{x,y})}[
    \mathbb{E}_{z \sim E(\mathbf{z|x})}[-\log{q(\mathbf{y|z})}]
  ] \\
  + \beta (\mathbb{E}_{\mathbf{x}\sim p(\mathbf{x})}[
    \text{KL}[E(\mathbf{z|x})||r(\mathbf{z})]
  ]-I_c)
  \}
\]`

ここで $\beta$ はラグランジュ乗数

+++

- 過学習しにくい
- Adversarial examplesに頑強

+++

## 教師あり学習

- 予測するモデルを学習し真のモデルへ近づける手法

__クロスエントロピー__

`\[
\min_{q} \mathbb{E}_{\mathbf{x, y}\sim p(\mathbf{x, y})}[-\log q(\mathbf{y|x})]
\]`

---

# Variational Discriminator Bottleneck

---

## GANの復習

__定式__

`\[
\max_{G} \min_{D} \mathbb{E}_{\mathbf{x}\sim p^* (\mathbf{x})}[-\log(D(\mathbf{x}))] \\
+ \mathbb{E}_{\mathbf{x}\sim G(\mathbf{x})}[-\log(1-D(\mathbf{x}))]
\]`

+++

##  Information Bottleneck法のGANへの応用

- 入力情報 $X$ に対しボトルネックと課す
   - encoder `\(E:\mathbf{z}\sim E(\mathbf{z|x})\)` を導入
   - 相互情報量 `\(I(X, Z)\)` を `\(I_c\)` に制限する

![VDB](VariationalDiscriminatorBottleneck/assets/VDB.png)

+++

- `\(\tilde{p} = \frac{1}{2} p^* + \frac{1}{2}G\)` として

__目的函数__

`\[
\begin{aligned}
J(D,E) &= \\
& \min_{D, E} \max_{\beta \geq 0 }
\mathbb{E}_{\mathbf{x}\sim p^* (\mathbf{x})}[
  \mathbb{E}_{\mathbf{z}\sim E(\mathbf{z|x})}[
    -\log(D(\mathbf{z}))
  ]
] \\
+ \mathbb{E}_{\mathbf{x}\sim G(\mathbf{x})}[
  \mathbb{E}_{\mathbf{z}\sim E(\mathbf{z|x})}[
    -\log(1-D(\mathbf{z}))
  ]
] \\
& + \beta (
\mathbb{E}_{\mathbf{x}\sim \tilde{p} (\mathbf{x})}[
  \text{KL}[E(\mathbf{z|x})||r(\mathbf{z})]
] - I_c
)
\end{aligned}
\]`

+++

## $I_c$ の効果

![Effects of `\(I_c\)` ](VariationalDiscriminatorBottleneck/assets/effectivesofIc.png)

+++

### 更新式

`\[
\begin{aligned}
& D, E \leftarrow \arg \min_{D,E} {\cal L}(D, E, \beta) \\
& \beta \leftarrow \max(0, \beta + \alpha _{\beta }(
  \mathbb{E}_{\mathbf{x}\sim \tilde{p} (\mathbf{x})}[
    \text{KL}[E(\mathbf{z|x})||r(\mathbf{z})]
  ] - I_c
  ))
\end{aligned}
\]`

ここで

`\[
\begin{aligned}
{\cal L}(D, E, \beta)　&=
\mathbb{E}_{\mathbf{x}\sim p^* (\mathbf{x})}[
  \mathbb{E}_{\mathbf{z}\sim E(\mathbf{z|x})}[
    -\log(D(\mathbf{z}))
  ]
]\\
&
+ \mathbb{E}_{\mathbf{x}\sim G(\mathbf{x})}[
  \mathbb{E}_{\mathbf{z}\sim E(\mathbf{z|x})}[
    -\log(1-D(\mathbf{z}))
  ]
]\\
&
+ \beta (
\mathbb{E}_{\mathbf{x}\sim \tilde{p} (\mathbf{x})}[
  \text{KL}[E(\mathbf{z|x})||r(\mathbf{z})]
] - I_c
), \\

\alpha_{\beta} &: \text{Stepsize for dual variable in dual gradient descent}
\end{aligned}
\]`

---

- 模倣学習
- 逆強化学習

への応用の理論的解説は省略
(GANとほぼ同じ議論になるため)

---

## EXPERIMENTS

- Image Generation
- Imitation Learning
- Inverse Reinforce Learning

---

## Image Generation

+++

## Learning model

- Gradient strategy: RMSProp (with a fixed learning rate)
- `\(\alpha_{\beta} = 10^{-5}\)`

![Generator](VariationalDiscriminatorBottleneck/assets/Generator_VDB.png)

![Discriminator](VariationalDiscriminatorBottleneck/assets/Discriminator_VDB.png)

+++

## Result

![image_generation_FID](VariationalDiscriminatorBottleneck/assets/image_generation_FID.png)

![image_generation](VariationalDiscriminatorBottleneck/assets/image_generation.png)
