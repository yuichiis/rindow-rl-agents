# Action Masks

Class: `Rindow\RL\Agents\Util\ActionMask`

`ActionMask::hasAny($la,$mask): bool` reduces a Boolean mask on its current
backend and reports whether at least one action is enabled.

For neural discrete agents, masks have shape `[numActions]` during inference and
`[batch,numActions]` during updates. Disabled logits or Q values are replaced by
a large negative finite value before softmax, maximum, or argmax. An all-false
mask is rejected because no valid action can be selected.

Dictionary observations configure masks with both `stateField` and
`actionMaskField`. Replay storage records the next-state mask; rollout storage
records the mask used to select each stored action.
