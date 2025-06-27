# observation
- envでの数値spacesをDiscreteを許すか、それともすべてBoxを強制するか悩むどころ
- すべてBoxに統一すれば、Estimator::getActionValues()のインターフェースのstatesは(batches,dims)に統一できる
- Discreteを許すとstatesは(batches)の場合を許す事になる。
- gym.Envを見ると、toy_text.cliffwalkingでDiscreteを使っていた。したがってBox統一という縛りは無いらしい。
- ほとんどのobservationはBoxのfloat型。
- これを入力とするのはEstimator::getActionValues()のstatesで、NNの入力値になる。
- ValueTableの場合はint型入力。getActionValues()の入力は(batches)とする。
- QNetworkの場合はfloat型入力。getActionValues()の入力は(batches,dims...)とする。
- MazeはDiscreteでint型となるのでQNetworkの場合はcustomStateFunctionでfloat型の(batches,1)に変換する。

# actions
- ActionのSpacesはDiscreteの場合とBoxの場合があるのは仕方ない。
- Discreteの場合は(batches)、Boxの場合は(batches,action1,action2,....)となる。
- MultiDiscreteは今回はサポートしない。
- Geminiに聞いたところMultiDiscreteは各出力を別の出力としてネットワークを作る必要があり複雑になるとの事。
- 1ボタン4方向スティックなどの複数のアクションの組み合わせを2x4の8通りとして扱い場合もあるがコマンドの数で爆発的に組み合わせが増えるので余りよくないという話。

# sample()
- Estimator::sample()が現在ランダムに選択した後のactionを返すようになっている。
- しかしEstimator::getActionValues()はActionValues(行動価値)を返すので、shapeもdtypeも違う。
- 役割分担を考えるならsample()では発生確率を返すべき。そうすればshapeも一致するしdtypeもfloat32になり統一される。
- ただし、行動価値は確率ではないので依然として意味の統一は完全にはされていない。
- Actionコマンドを複数種類必要とする場合も考える必要がある。今は出来てるかどうかわからない。
- メソッド名を変えるべきProbabilitiesとか。なぜならSpaces.Discrete.sample()とかがあるから。

# ActorCriticNetwork
- ActorCriticNetworkはまだ書きかけ状態。
- 2つ以上のアクションコマンドを同時に返す設計になっている。

# QNetwork
- Statesは複数変数の入力を許している。
- 出力の行動価値は1アクションコマンドにしか対応していない。
- 出力は(batches,numActions)

# RandomCategorical()
- Random::RandomCategorical()でバッチ処理に加えて複数アクションコマンドに対応する必要がある。
- 今はEstimatorのsample()から直接呼ばれているが、Policy::actions()の中で呼ぶように変更するべき


