"�D
uHostFlushSummaryWriter"FlushSummaryWriter(1ffff&\�@9ffff&\�@Affff&\�@Iffff&\�@a�����?i�����?�Unknown�
BHostIDLE"IDLE1     `�@A     `�@a���r�J�?i��?���?�Unknown
}HostMatMul")gradient_tape/sequential_1/dense_2/MatMul(1fffffFy@9fffffFy@AfffffFy@IfffffFy@a����%�?i'�N#��?�Unknown
sHost_FusedMatMul"sequential_1/dense_2/Relu(1fffff6s@9fffff6s@Afffff6s@Ifffff6s@a"E���l�?iPB�%�=�?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1     @i@9     @i@A     @i@I     @i@a�X�?i��!��?�Unknown
�HostGreaterEqual"+sequential_1/dropout_1/dropout/GreaterEqual(1�����|c@9�����|c@A�����|c@I�����|c@a��T�s��?ig�U���?�Unknown
^HostGatherV2"GatherV2(1fffff�`@9fffff�`@Afffff�`@Ifffff�`@a�$C6T3~?i�o��9�?�Unknown
�HostRandomUniform";sequential_1/dropout_1/dropout/random_uniform/RandomUniform(1     �]@9     �]@A     �]@I     �]@aM8���z?i!�v/�Q�?�Unknown
�	HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1333333E@9333333E@A333333E@I333333E@a�Q:c?i�;0e�?�Unknown
s
HostMul""sequential_1/dropout_1/dropout/Mul(1������C@9������C@A������C@I������C@a3BZn��a?iW@���v�?�Unknown
HostMatMul"+gradient_tape/sequential_1/dense_3/MatMul_1(1fffff&C@9fffff&C@Afffff&C@Ifffff&C@a0�D6^a?i�4�T��?�Unknown
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1fffff�A@9fffff�A@Afffff�A@Ifffff�A@aE�(�;`?i8]ǐ��?�Unknown
vHost_FusedMatMul"sequential_1/dense_3/BiasAdd(1333333;@9333333;@A333333;@I333333;@a��rZ�X?ikq�Q��?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1�����0@9�����0@A�����0@I�����0@axڸ��3M?i���M3��?�Unknown
iHostWriteSummary"WriteSummary(1ffffff-@9ffffff-@Affffff-@Iffffff-@aKc��۩J?i{��ݲ�?�Unknown�
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1333333+@9333333+@A333333+@I333333+@a��rZ�H?i/�����?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1������*@9������*@A333333$@I333333$@a�$w�QB?i�;����?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1������#@9������#@A������#@I������#@a3BZn��A?i��M���?�Unknown
dHostDataset"Iterator::Model(1������*@9������*@A������#@I������#@a3BZn��A?ii�J���?�Unknown
�HostReluGrad"+gradient_tape/sequential_1/dense_2/ReluGrad(1      !@9      !@A      !@I      !@a��q��>?i
�[��?�Unknown
uHostCast"#sequential_1/dropout_1/dropout/Cast(1������ @9������ @A������ @I������ @a����>?i�����?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1������@9������@A������@I������@a����<?i�	����?�Unknown
qHostSoftmax"sequential_1/dense_3/Softmax(1333333@9333333@A333333@I333333@aUޅ�K<?i�̙=��?�Unknown
uHostMul"$sequential_1/dropout_1/dropout/Mul_1(1333333@9333333@A333333@I333333@am(pl{:?i��'����?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������"@9������"@Affffff@Iffffff@a)�����9?i`'����?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1�����L<@9�����L<@A      @I      @a������3?i93C��?�Unknown
eHost
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a�| .��0?i*�2Y��?�Unknown�
�HostBiasAddGrad"6gradient_tape/sequential_1/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ac�b�S0?i|17�c��?�Unknown
`HostGatherV2"
GatherV2_1(1������@9������@A������@I������@a�GI�t�/?iF�\b��?�Unknown
�HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@am2b��+?i3L�|��?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1������@9������@A������@I������@a��j�*?iۢ�e���?�Unknown
� HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff
@9ffffff�?Affffff
@Iffffff�?a����W�'?iJ�U{<��?�Unknown
Z!HostArgMax"ArgMax(1������@9������@A������@I������@aw���#?iH3��u��?�Unknown
l"HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@aw���#?iFt�X���?�Unknown
�#HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a�| .�� ?iNVX���?�Unknown
[$HostAddV2"Adam/add(1������@9������@A������@I������@a�GI�t�?i�ྻ���?�Unknown
`%HostDivNoNan"
div_no_nan(1������@9������@A������@I������@a�GI�t�?i�jd���?�Unknown
�&HostReadVariableOp"*sequential_1/dense_3/MatMul/ReadVariableOp(1������ @9������ @A������ @I������ @a�Q�x?io�<���?�Unknown
�'HostMul"0gradient_tape/sequential_1/dropout_1/dropout/Mul(1       @9       @A       @I       @a=�Y[�?i>xG���?�Unknown
�(HostMul"2gradient_tape/sequential_1/dropout_1/dropout/Mul_2(1       @9       @A       @I       @a=�Y[�?iSR?}��?�Unknown
�)HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1       @9       @A       @I       @a=�Y[�?i�-]ke��?�Unknown
Y*HostPow"Adam/Pow(1�������?9�������?A�������?I�������?a��j�?i0Y�_6��?�Unknown
�+HostReadVariableOp"+sequential_1/dense_3/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a��j�?i��=T��?�Unknown
�,HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1�������?9�������?A�������?I�������?a��j�?iد�H���?�Unknown
V-HostSum"Sum_2(1333333�?9333333�?A333333�?I333333�?a��rZ�?in�P����?�Unknown
t.HostReadVariableOp"Adam/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a�{��7?iG�%^W��?�Unknown
[/HostPow"
Adam/Pow_1(1      �?9      �?A      �?I      �?a.k�!�?ib#.��?�Unknown
t0HostAssignAddVariableOp"AssignAddVariableOp(1      �?9      �?A      �?I      �?a.k�!�?i}G6����?�Unknown
v1HostCast"$sparse_categorical_crossentropy/Cast(1ffffff�?9ffffff�?Affffff�?Iffffff�?a^��Y�P?i�q%V��?�Unknown
v2HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1�������?9�������?A�������?I�������?a���-�?i{�����?�Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_2(1333333�?9333333�?A333333�?I333333�?a�U��i?i^�~\x��?�Unknown
�4HostReadVariableOp"+sequential_1/dense_2/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�U��i?iA����?�Unknown
~5HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1�������?9�������?A�������?I�������?a�GI�t�?if��[���?�Unknown
v6HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1�������?9�������?A�������?I�������?a�GI�t�?i�L���?�Unknown
�7HostReadVariableOp"*sequential_1/dense_2/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a�GI�t�?i������?�Unknown
o8HostReadVariableOp"Adam/ReadVariableOp(1      �?9      �?A      �?I      �?a=�Y[�?i�����?�Unknown
�9HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1      �?9      �?A      �?I      �?a=�Y[�?i���j��?�Unknown
X:HostEqual"Equal(1�������?9�������?A�������?I�������?a��j�
?i*�e���?�Unknown
u;HostReadVariableOp"div_no_nan/ReadVariableOp(1�������?9�������?A�������?I�������?a�{��7?i�DD0��?�Unknown
]<HostCast"Adam/Cast_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a^��Y�P?iE&ↁ��?�Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a^��Y�P?it�����?�Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?a^��Y�P?i��$��?�Unknown
X?HostCast"Cast_2(1333333�?9333333�?A333333�?I333333�?a�U��i?i��i��?�Unknown
X@HostCast"Cast_3(1333333�?9333333�?A333333�?I333333�?a�U��i?i��Y���?�Unknown
bAHostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?a�U��i?i�� ���?�Unknown
aBHostIdentity"Identity(1      �?9      �?A      �?I      �?a=�Y[��>i�ԏ/��?�Unknown�
�CHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      �?9      �?A      �?I      �?a=�Y[��>i^��i��?�Unknown
TDHostMul"Mul(1333333�?9333333�?A333333�?I333333�?a�U��i�>i������?�Unknown
wEHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a�U��i�>iЙb����?�Unknown
yFHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a�U��i�>i	�ʐ���?�Unknown
vGHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a�{��7�>i�Pe����?�Unknown
wHHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a�{��7�>i�������?�Unknown*�D
uHostFlushSummaryWriter"FlushSummaryWriter(1ffff&\�@9ffff&\�@Affff&\�@Iffff&\�@a����i�?i����i�?�Unknown�
}HostMatMul")gradient_tape/sequential_1/dense_2/MatMul(1fffffFy@9fffffFy@AfffffFy@IfffffFy@aϸ=�?i���C�?�Unknown
sHost_FusedMatMul"sequential_1/dense_2/Relu(1fffff6s@9fffff6s@Afffff6s@Ifffff6s@a������?i�:}����?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1     @i@9     @i@A     @i@I     @i@a_����6�?i:�T�U�?�Unknown
�HostGreaterEqual"+sequential_1/dropout_1/dropout/GreaterEqual(1�����|c@9�����|c@A�����|c@I�����|c@aԖ��� �?i�l�a���?�Unknown
^HostGatherV2"GatherV2(1fffff�`@9fffff�`@Afffff�`@Ifffff�`@a����?i  7>P��?�Unknown
�HostRandomUniform";sequential_1/dropout_1/dropout/random_uniform/RandomUniform(1     �]@9     �]@A     �]@I     �]@a\|�q��?i+#,1�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1333333E@9333333E@A333333E@I333333E@apI؁_�f?ib��H�?�Unknown
s	HostMul""sequential_1/dropout_1/dropout/Mul(1������C@9������C@A������C@I������C@a�0l4�e?i�o�k%]�?�Unknown

HostMatMul"+gradient_tape/sequential_1/dense_3/MatMul_1(1fffff&C@9fffff&C@Afffff&C@Ifffff&C@a�ɥd?i]�+�q�?�Unknown
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1fffff�A@9fffff�A@Afffff�A@Ifffff�A@a(V)��Jc?i�>A��?�Unknown
vHost_FusedMatMul"sequential_1/dense_3/BiasAdd(1333333;@9333333;@A333333;@I333333;@a��-$�P]?i�U�t���?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1�����0@9�����0@A�����0@I�����0@a0�k6ZQ?ic��i��?�Unknown
iHostWriteSummary"WriteSummary(1ffffff-@9ffffff-@Affffff-@Iffffff-@a�H��ݯO?i��|�U��?�Unknown�
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1333333+@9333333+@A333333+@I333333+@a��-$�PM?i_�E����?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1������*@9������*@A333333$@I333333$@a�t�u�E?i����?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1������#@9������#@A������#@I������#@a�0l4�E?i�	�c��?�Unknown
dHostDataset"Iterator::Model(1������*@9������*@A������#@I������#@a�0l4�E?i�$D���?�Unknown
�HostReluGrad"+gradient_tape/sequential_1/dense_2/ReluGrad(1      !@9      !@A      !@I      !@a,����RB?i�ˁ�?��?�Unknown
uHostCast"#sequential_1/dropout_1/dropout/Cast(1������ @9������ @A������ @I������ @a��Ac+�A?i'�Z����?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1������@9������@A������@I������@a�u�<pA?i�i����?�Unknown
qHostSoftmax"sequential_1/dense_3/Softmax(1333333@9333333@A333333@I333333@ak��rA�@?i�vƩ.��?�Unknown
uHostMul"$sequential_1/dropout_1/dropout/Mul_1(1333333@9333333@A333333@I333333@a����x??i����?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������"@9������"@Affffff@Iffffff@aR�>��>?i��">���?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1�����L<@9�����L<@A      @I      @a�U���7?i��w���?�Unknown
eHost
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a�[z��3?i :��b��?�Unknown�
�HostBiasAddGrad"6gradient_tape/sequential_1/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a� �rf3?i#�i���?�Unknown
`HostGatherV2"
GatherV2_1(1������@9������@A������@I������@a�S�2?iŎ�l.��?�Unknown
�HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a3���a0?i#
�:��?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1������@9������@A������@I������@a���qQ
/?i��!N+��?�Unknown
�HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff
@9ffffff�?Affffff
@Iffffff�?a>�w�t,?i9p!����?�Unknown
Z HostArgMax"ArgMax(1������@9������@A������@I������@a�O3�G'?in��g��?�Unknown
l!HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@a�O3�G'?i�ć���?�Unknown
�"HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a�[z��#?iU�����?�Unknown
[#HostAddV2"Adam/add(1������@9������@A������@I������@a�S�"?i�� VH��?�Unknown
`$HostDivNoNan"
div_no_nan(1������@9������@A������@I������@a�S�"?i�0v�w��?�Unknown
�%HostReadVariableOp"*sequential_1/dense_3/MatMul/ReadVariableOp(1������ @9������ @A������ @I������ @a�,Z"?i������?�Unknown
�&HostMul"0gradient_tape/sequential_1/dropout_1/dropout/Mul(1       @9       @A       @I       @a��8�>!?iwc	w���?�Unknown
�'HostMul"2gradient_tape/sequential_1/dropout_1/dropout/Mul_2(1       @9       @A       @I       @a��8�>!?i��`���?�Unknown
�(HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1       @9       @A       @I       @a��8�>!?i�*�J���?�Unknown
Y)HostPow"Adam/Pow(1�������?9�������?A�������?I�������?a���qQ
?ie�u����?�Unknown
�*HostReadVariableOp"+sequential_1/dense_3/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a���qQ
?i3D����?�Unknown
�+HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1�������?9�������?A�������?I�������?a���qQ
?iьB���?�Unknown
V,HostSum"Sum_2(1333333�?9333333�?A333333�?I333333�?a��-$�P?in�eɨ��?�Unknown
t-HostReadVariableOp"Adam/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?aЍ��d�?iz������?�Unknown
[.HostPow"
Adam/Pow_1(1      �?9      �?A      �?I      �?a�tU���?i&� tT��?�Unknown
t/HostAssignAddVariableOp"AssignAddVariableOp(1      �?9      �?A      �?I      �?a�tU���?i�=uc#��?�Unknown
v0HostCast"$sparse_categorical_crossentropy/Cast(1ffffff�?9ffffff�?Affffff�?Iffffff�?a\�;x$?i7����?�Unknown
v1HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1�������?9�������?A�������?I�������?a9C}�k?i�Fߗ��?�Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_2(1333333�?9333333�?A333333�?I333333�?a\*���?i���k=��?�Unknown
�3HostReadVariableOp"+sequential_1/dense_2/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a\*���?i� ����?�Unknown
~4HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1�������?9�������?A�������?I�������?a�S�?iB?��z��?�Unknown
v5HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1�������?9�������?A�������?I�������?a�S�?ik�Uy��?�Unknown
�6HostReadVariableOp"*sequential_1/dense_2/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a�S�?i�y :���?�Unknown
o7HostReadVariableOp"Adam/ReadVariableOp(1      �?9      �?A      �?I      �?a��8�>?i\��.4��?�Unknown
�8HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1      �?9      �?A      �?I      �?a��8�>?i$��#���?�Unknown
X9HostEqual"Equal(1�������?9�������?A�������?I�������?a���qQ
?i��6M:��?�Unknown
u:HostReadVariableOp"div_no_nan/ReadVariableOp(1�������?9�������?A�������?I�������?aЍ��d�?i��ɪ���?�Unknown
];HostCast"Adam/Cast_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a\�;x$?i6�<	��?�Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a\�;x$?i�݋�i��?�Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?a\�;x$?i��l`���?�Unknown
X>HostCast"Cast_2(1333333�?9333333�?A333333�?I333333�?a\*���?i�Q�&��?�Unknown
X?HostCast"Cast_3(1333333�?9333333�?A333333�?I333333�?a\*���?i
���o��?�Unknown
b@HostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?a\*���?iOZ�����?�Unknown
aAHostIdentity"Identity(1      �?9      �?A      �?I      �?a��8�>?i3st���?�Unknown�
�BHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      �?9      �?A      �?I      �?a��8�>?i��L��?�Unknown
TCHostMul"Mul(1333333�?9333333�?A333333�?I333333�?a\*����>i9�v��?�Unknown
wDHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a\*����>i[n���?�Unknown
yEHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a\*����>i}R6����?�Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?aЍ��d��>i?)�h���?�Unknown
wGHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?aЍ��d��>i      �?�Unknown