„
Ł*Æ*
:
Add
x"T
y"T
z"T"
Ttype:
2	
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
>
DiagPart

input"T
diagonal"T"
Ttype:

2	
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

)
Exit	
data"T
output"T"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2

GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
!
LoopCond	
input


output

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	

M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
Qr

input"T
q"T
r"T"
full_matricesbool( "
Ttype:
2

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	

ResourceGather
resource
indices"Tindices
output"dtype"
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
/
Sign
x"T
y"T"
Ttype:

2	
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
9
TensorArraySizeV3

handle
flow_in
size
Ž
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized
"serve*1.12.22v1.12.0-35-gcf74798993ĶŹ


word_inputPlaceholder*%
shape:’’’’’’’’’’’’’’’’’’*
dtype0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
W
masking/NotEqual/yConst*
valueB
 *  A*
dtype0*
_output_shapes
: 
w
masking/NotEqualNotEqual
word_inputmasking/NotEqual/y*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
h
masking/Any/reduction_indicesConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

masking/AnyAnymasking/NotEqualmasking/Any/reduction_indices*'
_output_shapes
:’’’’’’’’’*
	keep_dims(*

Tidx0
r
masking/CastCastmasking/Any*

SrcT0
*
Truncate( *'
_output_shapes
:’’’’’’’’’*

DstT0
g
masking/mulMul
word_inputmasking/Cast*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
Y
masking/NotEqual_1/yConst*
valueB
 *  A*
dtype0*
_output_shapes
: 
{
masking/NotEqual_1NotEqual
word_inputmasking/NotEqual_1/y*0
_output_shapes
:’’’’’’’’’’’’’’’’’’*
T0
j
masking/Any_1/reduction_indicesConst*
_output_shapes
: *
valueB :
’’’’’’’’’*
dtype0

masking/Any_1Anymasking/NotEqual_1masking/Any_1/reduction_indices*#
_output_shapes
:’’’’’’’’’*
	keep_dims( *

Tidx0
Æ
5embedding/embeddings/Initializer/random_uniform/shapeConst*'
_class
loc:@embedding/embeddings*
valueB"bz  ,  *
dtype0*
_output_shapes
:
”
3embedding/embeddings/Initializer/random_uniform/minConst*'
_class
loc:@embedding/embeddings*
valueB
 *ĶĢL½*
dtype0*
_output_shapes
: 
”
3embedding/embeddings/Initializer/random_uniform/maxConst*'
_class
loc:@embedding/embeddings*
valueB
 *ĶĢL=*
dtype0*
_output_shapes
: 

=embedding/embeddings/Initializer/random_uniform/RandomUniformRandomUniform5embedding/embeddings/Initializer/random_uniform/shape*

seed *
T0*'
_class
loc:@embedding/embeddings*
seed2 *
dtype0*!
_output_shapes
:āō¬
ī
3embedding/embeddings/Initializer/random_uniform/subSub3embedding/embeddings/Initializer/random_uniform/max3embedding/embeddings/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@embedding/embeddings

3embedding/embeddings/Initializer/random_uniform/mulMul=embedding/embeddings/Initializer/random_uniform/RandomUniform3embedding/embeddings/Initializer/random_uniform/sub*
T0*'
_class
loc:@embedding/embeddings*!
_output_shapes
:āō¬
õ
/embedding/embeddings/Initializer/random_uniformAdd3embedding/embeddings/Initializer/random_uniform/mul3embedding/embeddings/Initializer/random_uniform/min*!
_output_shapes
:āō¬*
T0*'
_class
loc:@embedding/embeddings
Į
embedding/embeddingsVarHandleOp*
_output_shapes
: *%
shared_nameembedding/embeddings*'
_class
loc:@embedding/embeddings*
	container *
shape:āō¬*
dtype0
y
5embedding/embeddings/IsInitialized/VarIsInitializedOpVarIsInitializedOpembedding/embeddings*
_output_shapes
: 
¬
embedding/embeddings/AssignAssignVariableOpembedding/embeddings/embedding/embeddings/Initializer/random_uniform*'
_class
loc:@embedding/embeddings*
dtype0
©
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*'
_class
loc:@embedding/embeddings*
dtype0*!
_output_shapes
:āō¬
}
embedding/CastCastmasking/mul*0
_output_shapes
:’’’’’’’’’’’’’’’’’’*

DstT0*

SrcT0*
Truncate( 

.embedding/embedding_lookup/Read/ReadVariableOpReadVariableOpembedding/embeddings*
dtype0*!
_output_shapes
:āō¬

#embedding/embedding_lookup/IdentityIdentity.embedding/embedding_lookup/Read/ReadVariableOp*
T0*!
_output_shapes
:āō¬

embedding/embedding_lookupResourceGatherembedding/embeddingsembedding/Cast*A
_class7
53loc:@embedding/embedding_lookup/Read/ReadVariableOp*
dtype0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’¬*
Tindices0*
validate_indices(
Š
%embedding/embedding_lookup/Identity_1Identityembedding/embedding_lookup*A
_class7
53loc:@embedding/embedding_lookup/Read/ReadVariableOp*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’¬*
T0

%embedding/embedding_lookup/Identity_2Identity%embedding/embedding_lookup/Identity_1*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’¬


char_inputPlaceholder*
dtype0*=
_output_shapes+
):'’’’’’’’’’’’’’’’’’’’’’’’’’’’*2
shape):'’’’’’’’’’’’’’’’’’’’’’’’’’’’
V
lambda/ShapeShape
char_input*
T0*
out_type0*
_output_shapes
:
m
lambda/strided_slice/stackConst*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:
f
lambda/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
f
lambda/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0

lambda/strided_sliceStridedSlicelambda/Shapelambda/strided_slice/stacklambda/strided_slice/stack_1lambda/strided_slice/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
a
lambda/Reshape/shape/0Const*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

lambda/Reshape/shapePacklambda/Reshape/shape/0lambda/strided_slice*
T0*

axis *
N*
_output_shapes
:

lambda/ReshapeReshape
char_inputlambda/Reshape/shape*
Tshape0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’*
T0
Y
masking_1/NotEqual/yConst*
valueB
 *  A*
dtype0*
_output_shapes
: 

masking_1/NotEqualNotEquallambda/Reshapemasking_1/NotEqual/y*0
_output_shapes
:’’’’’’’’’’’’’’’’’’*
T0
j
masking_1/Any/reduction_indicesConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

masking_1/AnyAnymasking_1/NotEqualmasking_1/Any/reduction_indices*'
_output_shapes
:’’’’’’’’’*
	keep_dims(*

Tidx0
v
masking_1/CastCastmasking_1/Any*

SrcT0
*
Truncate( *'
_output_shapes
:’’’’’’’’’*

DstT0
o
masking_1/mulMullambda/Reshapemasking_1/Cast*0
_output_shapes
:’’’’’’’’’’’’’’’’’’*
T0
[
masking_1/NotEqual_1/yConst*
valueB
 *  A*
dtype0*
_output_shapes
: 

masking_1/NotEqual_1NotEquallambda/Reshapemasking_1/NotEqual_1/y*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
l
!masking_1/Any_1/reduction_indicesConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

masking_1/Any_1Anymasking_1/NotEqual_1!masking_1/Any_1/reduction_indices*#
_output_shapes
:’’’’’’’’’*
	keep_dims( *

Tidx0
³
7embedding_1/embeddings/Initializer/random_uniform/shapeConst*
_output_shapes
:*)
_class
loc:@embedding_1/embeddings*
valueB"f   d   *
dtype0
„
5embedding_1/embeddings/Initializer/random_uniform/minConst*
_output_shapes
: *)
_class
loc:@embedding_1/embeddings*
valueB
 *ĶĢL½*
dtype0
„
5embedding_1/embeddings/Initializer/random_uniform/maxConst*
_output_shapes
: *)
_class
loc:@embedding_1/embeddings*
valueB
 *ĶĢL=*
dtype0

?embedding_1/embeddings/Initializer/random_uniform/RandomUniformRandomUniform7embedding_1/embeddings/Initializer/random_uniform/shape*
T0*)
_class
loc:@embedding_1/embeddings*
seed2 *
dtype0*
_output_shapes

:fd*

seed 
ö
5embedding_1/embeddings/Initializer/random_uniform/subSub5embedding_1/embeddings/Initializer/random_uniform/max5embedding_1/embeddings/Initializer/random_uniform/min*
_output_shapes
: *
T0*)
_class
loc:@embedding_1/embeddings

5embedding_1/embeddings/Initializer/random_uniform/mulMul?embedding_1/embeddings/Initializer/random_uniform/RandomUniform5embedding_1/embeddings/Initializer/random_uniform/sub*)
_class
loc:@embedding_1/embeddings*
_output_shapes

:fd*
T0
ś
1embedding_1/embeddings/Initializer/random_uniformAdd5embedding_1/embeddings/Initializer/random_uniform/mul5embedding_1/embeddings/Initializer/random_uniform/min*)
_class
loc:@embedding_1/embeddings*
_output_shapes

:fd*
T0
Ä
embedding_1/embeddingsVarHandleOp*
	container *
shape
:fd*
dtype0*
_output_shapes
: *'
shared_nameembedding_1/embeddings*)
_class
loc:@embedding_1/embeddings
}
7embedding_1/embeddings/IsInitialized/VarIsInitializedOpVarIsInitializedOpembedding_1/embeddings*
_output_shapes
: 
“
embedding_1/embeddings/AssignAssignVariableOpembedding_1/embeddings1embedding_1/embeddings/Initializer/random_uniform*)
_class
loc:@embedding_1/embeddings*
dtype0
¬
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*)
_class
loc:@embedding_1/embeddings*
dtype0*
_output_shapes

:fd

embedding_1/CastCastmasking_1/mul*

SrcT0*
Truncate( *0
_output_shapes
:’’’’’’’’’’’’’’’’’’*

DstT0

0embedding_1/embedding_lookup/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
dtype0*
_output_shapes

:fd

%embedding_1/embedding_lookup/IdentityIdentity0embedding_1/embedding_lookup/Read/ReadVariableOp*
_output_shapes

:fd*
T0

embedding_1/embedding_lookupResourceGatherembedding_1/embeddingsembedding_1/Cast*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d*
Tindices0*
validate_indices(*C
_class9
75loc:@embedding_1/embedding_lookup/Read/ReadVariableOp*
dtype0
Õ
'embedding_1/embedding_lookup/Identity_1Identityembedding_1/embedding_lookup*C
_class9
75loc:@embedding_1/embedding_lookup/Read/ReadVariableOp*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d*
T0

'embedding_1/embedding_lookup/Identity_2Identity'embedding_1/embedding_lookup/Identity_1*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d*
T0
d
"dropout/keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 

dropout/keras_learning_phasePlaceholderWithDefault"dropout/keras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 
|
dropout/cond/SwitchSwitchdropout/keras_learning_phasedropout/keras_learning_phase*
T0
*
_output_shapes
: : 
Y
dropout/cond/switch_tIdentitydropout/cond/Switch:1*
_output_shapes
: *
T0

W
dropout/cond/switch_fIdentitydropout/cond/Switch*
_output_shapes
: *
T0

_
dropout/cond/pred_idIdentitydropout/keras_learning_phase*
T0
*
_output_shapes
: 
{
dropout/cond/dropout/keep_probConst^dropout/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
}
dropout/cond/dropout/ShapeShape#dropout/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:

!dropout/cond/dropout/Shape/SwitchSwitch'embedding_1/embedding_lookup/Identity_2dropout/cond/pred_id*
T0*:
_class0
.,loc:@embedding_1/embedding_lookup/Identity_2*T
_output_shapesB
@:’’’’’’’’’’’’’’’’’’d:’’’’’’’’’’’’’’’’’’d

'dropout/cond/dropout/random_uniform/minConst^dropout/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

'dropout/cond/dropout/random_uniform/maxConst^dropout/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ć
1dropout/cond/dropout/random_uniform/RandomUniformRandomUniformdropout/cond/dropout/Shape*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d*
seed2 *

seed *
T0*
dtype0
”
'dropout/cond/dropout/random_uniform/subSub'dropout/cond/dropout/random_uniform/max'dropout/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
É
'dropout/cond/dropout/random_uniform/mulMul1dropout/cond/dropout/random_uniform/RandomUniform'dropout/cond/dropout/random_uniform/sub*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d*
T0
»
#dropout/cond/dropout/random_uniformAdd'dropout/cond/dropout/random_uniform/mul'dropout/cond/dropout/random_uniform/min*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d*
T0
£
dropout/cond/dropout/addAdddropout/cond/dropout/keep_prob#dropout/cond/dropout/random_uniform*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d
|
dropout/cond/dropout/FloorFloordropout/cond/dropout/add*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d*
T0
§
dropout/cond/dropout/divRealDiv#dropout/cond/dropout/Shape/Switch:1dropout/cond/dropout/keep_prob*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d*
T0

dropout/cond/dropout/mulMuldropout/cond/dropout/divdropout/cond/dropout/Floor*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d
~
dropout/cond/IdentityIdentitydropout/cond/Identity/Switch*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d

dropout/cond/Identity/SwitchSwitch'embedding_1/embedding_lookup/Identity_2dropout/cond/pred_id*:
_class0
.,loc:@embedding_1/embedding_lookup/Identity_2*T
_output_shapesB
@:’’’’’’’’’’’’’’’’’’d:’’’’’’’’’’’’’’’’’’d*
T0

dropout/cond/MergeMergedropout/cond/Identitydropout/cond/dropout/mul*
T0*
N*6
_output_shapes$
":’’’’’’’’’’’’’’’’’’d: 

,lstm/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
_class
loc:@lstm/kernel*
valueB"d     *
dtype0

*lstm/kernel/Initializer/random_uniform/minConst*
_class
loc:@lstm/kernel*
valueB
 *ßXą½*
dtype0*
_output_shapes
: 

*lstm/kernel/Initializer/random_uniform/maxConst*
_class
loc:@lstm/kernel*
valueB
 *ßXą=*
dtype0*
_output_shapes
: 
ć
4lstm/kernel/Initializer/random_uniform/RandomUniformRandomUniform,lstm/kernel/Initializer/random_uniform/shape*
_output_shapes
:	d*

seed *
T0*
_class
loc:@lstm/kernel*
seed2 *
dtype0
Ź
*lstm/kernel/Initializer/random_uniform/subSub*lstm/kernel/Initializer/random_uniform/max*lstm/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@lstm/kernel
Ż
*lstm/kernel/Initializer/random_uniform/mulMul4lstm/kernel/Initializer/random_uniform/RandomUniform*lstm/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@lstm/kernel*
_output_shapes
:	d
Ļ
&lstm/kernel/Initializer/random_uniformAdd*lstm/kernel/Initializer/random_uniform/mul*lstm/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@lstm/kernel*
_output_shapes
:	d
¤
lstm/kernelVarHandleOp*
_class
loc:@lstm/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
: *
shared_namelstm/kernel
g
,lstm/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOplstm/kernel*
_output_shapes
: 

lstm/kernel/AssignAssignVariableOplstm/kernel&lstm/kernel/Initializer/random_uniform*
_class
loc:@lstm/kernel*
dtype0

lstm/kernel/Read/ReadVariableOpReadVariableOplstm/kernel*
_class
loc:@lstm/kernel*
dtype0*
_output_shapes
:	d
°
5lstm/recurrent_kernel/Initializer/random_normal/shapeConst*(
_class
loc:@lstm/recurrent_kernel*
valueB"  d   *
dtype0*
_output_shapes
:
£
4lstm/recurrent_kernel/Initializer/random_normal/meanConst*(
_class
loc:@lstm/recurrent_kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
„
6lstm/recurrent_kernel/Initializer/random_normal/stddevConst*(
_class
loc:@lstm/recurrent_kernel*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Dlstm/recurrent_kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal5lstm/recurrent_kernel/Initializer/random_normal/shape*

seed *
T0*(
_class
loc:@lstm/recurrent_kernel*
seed2 *
dtype0*
_output_shapes
:	d

3lstm/recurrent_kernel/Initializer/random_normal/mulMulDlstm/recurrent_kernel/Initializer/random_normal/RandomStandardNormal6lstm/recurrent_kernel/Initializer/random_normal/stddev*
T0*(
_class
loc:@lstm/recurrent_kernel*
_output_shapes
:	d
õ
/lstm/recurrent_kernel/Initializer/random_normalAdd3lstm/recurrent_kernel/Initializer/random_normal/mul4lstm/recurrent_kernel/Initializer/random_normal/mean*
T0*(
_class
loc:@lstm/recurrent_kernel*
_output_shapes
:	d
Ī
$lstm/recurrent_kernel/Initializer/QrQr/lstm/recurrent_kernel/Initializer/random_normal*(
_class
loc:@lstm/recurrent_kernel*)
_output_shapes
:	d:dd*
full_matrices( *
T0
­
*lstm/recurrent_kernel/Initializer/DiagPartDiagPart&lstm/recurrent_kernel/Initializer/Qr:1*
T0*(
_class
loc:@lstm/recurrent_kernel*
_output_shapes
:d
©
&lstm/recurrent_kernel/Initializer/SignSign*lstm/recurrent_kernel/Initializer/DiagPart*
_output_shapes
:d*
T0*(
_class
loc:@lstm/recurrent_kernel
Ī
%lstm/recurrent_kernel/Initializer/mulMul$lstm/recurrent_kernel/Initializer/Qr&lstm/recurrent_kernel/Initializer/Sign*
T0*(
_class
loc:@lstm/recurrent_kernel*
_output_shapes
:	d
¼
Alstm/recurrent_kernel/Initializer/matrix_transpose/transpose/permConst*(
_class
loc:@lstm/recurrent_kernel*
valueB"       *
dtype0*
_output_shapes
:

<lstm/recurrent_kernel/Initializer/matrix_transpose/transpose	Transpose%lstm/recurrent_kernel/Initializer/mulAlstm/recurrent_kernel/Initializer/matrix_transpose/transpose/perm*
T0*(
_class
loc:@lstm/recurrent_kernel*
_output_shapes
:	d*
Tperm0
Ŗ
/lstm/recurrent_kernel/Initializer/Reshape/shapeConst*(
_class
loc:@lstm/recurrent_kernel*
valueB"d     *
dtype0*
_output_shapes
:

)lstm/recurrent_kernel/Initializer/ReshapeReshape<lstm/recurrent_kernel/Initializer/matrix_transpose/transpose/lstm/recurrent_kernel/Initializer/Reshape/shape*(
_class
loc:@lstm/recurrent_kernel*
Tshape0*
_output_shapes
:	d*
T0

)lstm/recurrent_kernel/Initializer/mul_1/xConst*(
_class
loc:@lstm/recurrent_kernel*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ų
'lstm/recurrent_kernel/Initializer/mul_1Mul)lstm/recurrent_kernel/Initializer/mul_1/x)lstm/recurrent_kernel/Initializer/Reshape*
T0*(
_class
loc:@lstm/recurrent_kernel*
_output_shapes
:	d
Ā
lstm/recurrent_kernelVarHandleOp*
	container *
shape:	d*
dtype0*
_output_shapes
: *&
shared_namelstm/recurrent_kernel*(
_class
loc:@lstm/recurrent_kernel
{
6lstm/recurrent_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOplstm/recurrent_kernel*
_output_shapes
: 
§
lstm/recurrent_kernel/AssignAssignVariableOplstm/recurrent_kernel'lstm/recurrent_kernel/Initializer/mul_1*(
_class
loc:@lstm/recurrent_kernel*
dtype0
Ŗ
)lstm/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/recurrent_kernel*(
_class
loc:@lstm/recurrent_kernel*
dtype0*
_output_shapes
:	d

lstm/bias/Initializer/zerosConst*
_output_shapes
:d*
_class
loc:@lstm/bias*
valueBd*    *
dtype0

lstm/bias/Initializer/onesConst*
_class
loc:@lstm/bias*
valueBd*  ?*
dtype0*
_output_shapes
:d

lstm/bias/Initializer/zeros_1Const*
_output_shapes	
:Č*
_class
loc:@lstm/bias*
valueBČ*    *
dtype0

!lstm/bias/Initializer/concat/axisConst*
_class
loc:@lstm/bias*
value	B : *
dtype0*
_output_shapes
: 
ü
lstm/bias/Initializer/concatConcatV2lstm/bias/Initializer/zeroslstm/bias/Initializer/oneslstm/bias/Initializer/zeros_1!lstm/bias/Initializer/concat/axis*
T0*
_class
loc:@lstm/bias*
N*
_output_shapes	
:*

Tidx0

	lstm/biasVarHandleOp*
_output_shapes
: *
shared_name	lstm/bias*
_class
loc:@lstm/bias*
	container *
shape:*
dtype0
c
*lstm/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp	lstm/bias*
_output_shapes
: 
x
lstm/bias/AssignAssignVariableOp	lstm/biaslstm/bias/Initializer/concat*
_class
loc:@lstm/bias*
dtype0

lstm/bias/Read/ReadVariableOpReadVariableOp	lstm/bias*
_class
loc:@lstm/bias*
dtype0*
_output_shapes	
:
\

lstm/ShapeShapedropout/cond/Merge*
out_type0*
_output_shapes
:*
T0
b
lstm/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
d
lstm/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0

lstm/strided_sliceStridedSlice
lstm/Shapelstm/strided_slice/stacklstm/strided_slice/stack_1lstm/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
R
lstm/zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: 
\
lstm/zeros/mulMullstm/strided_slicelstm/zeros/mul/y*
T0*
_output_shapes
: 
T
lstm/zeros/Less/yConst*
value
B :č*
dtype0*
_output_shapes
: 
[
lstm/zeros/LessLesslstm/zeros/mullstm/zeros/Less/y*
_output_shapes
: *
T0
U
lstm/zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: 
|
lstm/zeros/packedPacklstm/strided_slicelstm/zeros/packed/1*
T0*

axis *
N*
_output_shapes
:
U
lstm/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
{

lstm/zerosFilllstm/zeros/packedlstm/zeros/Const*'
_output_shapes
:’’’’’’’’’d*
T0*

index_type0
T
lstm/zeros_1/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: 
`
lstm/zeros_1/mulMullstm/strided_slicelstm/zeros_1/mul/y*
T0*
_output_shapes
: 
V
lstm/zeros_1/Less/yConst*
value
B :č*
dtype0*
_output_shapes
: 
a
lstm/zeros_1/LessLesslstm/zeros_1/mullstm/zeros_1/Less/y*
T0*
_output_shapes
: 
W
lstm/zeros_1/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: 

lstm/zeros_1/packedPacklstm/strided_slicelstm/zeros_1/packed/1*
_output_shapes
:*
T0*

axis *
N
W
lstm/zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

lstm/zeros_1Filllstm/zeros_1/packedlstm/zeros_1/Const*
T0*

index_type0*'
_output_shapes
:’’’’’’’’’d
h
lstm/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:

lstm/transpose	Transposedropout/cond/Mergelstm/transpose/perm*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d*
Tperm0
Z
lstm/Shape_1Shapelstm/transpose*
T0*
out_type0*
_output_shapes
:
d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0
f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
f
lstm/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

lstm/strided_slice_1StridedSlicelstm/Shape_1lstm/strided_slice_1/stacklstm/strided_slice_1/stack_1lstm/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
d
lstm/strided_slice_2/stackConst*
_output_shapes
:*
valueB: *
dtype0
f
lstm/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
f
lstm/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Æ
lstm/strided_slice_2StridedSlicelstm/transposelstm/strided_slice_2/stacklstm/strided_slice_2/stack_1lstm/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *'
_output_shapes
:’’’’’’’’’d*
T0*
Index0
`
lstm/ReadVariableOpReadVariableOplstm/kernel*
dtype0*
_output_shapes
:	d
k
lstm/strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:
m
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
valueB"    d   *
dtype0
m
lstm/strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
«
lstm/strided_slice_3StridedSlicelstm/ReadVariableOplstm/strided_slice_3/stacklstm/strided_slice_3/stack_1lstm/strided_slice_3/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:dd*
Index0*
T0

lstm/MatMulMatMullstm/strided_slice_2lstm/strided_slice_3*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
b
lstm/ReadVariableOp_1ReadVariableOplstm/kernel*
_output_shapes
:	d*
dtype0
k
lstm/strided_slice_4/stackConst*
valueB"    d   *
dtype0*
_output_shapes
:
m
lstm/strided_slice_4/stack_1Const*
valueB"    Č   *
dtype0*
_output_shapes
:
m
lstm/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
­
lstm/strided_slice_4StridedSlicelstm/ReadVariableOp_1lstm/strided_slice_4/stacklstm/strided_slice_4/stack_1lstm/strided_slice_4/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:dd*
Index0*
T0*
shrink_axis_mask 

lstm/MatMul_1MatMullstm/strided_slice_2lstm/strided_slice_4*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
b
lstm/ReadVariableOp_2ReadVariableOplstm/kernel*
_output_shapes
:	d*
dtype0
k
lstm/strided_slice_5/stackConst*
valueB"    Č   *
dtype0*
_output_shapes
:
m
lstm/strided_slice_5/stack_1Const*
valueB"    ,  *
dtype0*
_output_shapes
:
m
lstm/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
­
lstm/strided_slice_5StridedSlicelstm/ReadVariableOp_2lstm/strided_slice_5/stacklstm/strided_slice_5/stack_1lstm/strided_slice_5/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:dd*
Index0*
T0

lstm/MatMul_2MatMullstm/strided_slice_2lstm/strided_slice_5*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( *
T0
b
lstm/ReadVariableOp_3ReadVariableOplstm/kernel*
dtype0*
_output_shapes
:	d
k
lstm/strided_slice_6/stackConst*
valueB"    ,  *
dtype0*
_output_shapes
:
m
lstm/strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
m
lstm/strided_slice_6/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
­
lstm/strided_slice_6StridedSlicelstm/ReadVariableOp_3lstm/strided_slice_6/stacklstm/strided_slice_6/stack_1lstm/strided_slice_6/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:dd*
Index0*
T0

lstm/MatMul_3MatMullstm/strided_slice_2lstm/strided_slice_6*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
\
lstm/ReadVariableOp_4ReadVariableOp	lstm/bias*
dtype0*
_output_shapes	
:
d
lstm/strided_slice_7/stackConst*
_output_shapes
:*
valueB: *
dtype0
f
lstm/strided_slice_7/stack_1Const*
valueB:d*
dtype0*
_output_shapes
:
f
lstm/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
©
lstm/strided_slice_7StridedSlicelstm/ReadVariableOp_4lstm/strided_slice_7/stacklstm/strided_slice_7/stack_1lstm/strided_slice_7/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
:d*
T0*
Index0

lstm/BiasAddBiasAddlstm/MatMullstm/strided_slice_7*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’d
\
lstm/ReadVariableOp_5ReadVariableOp	lstm/bias*
dtype0*
_output_shapes	
:
d
lstm/strided_slice_8/stackConst*
valueB:d*
dtype0*
_output_shapes
:
g
lstm/strided_slice_8/stack_1Const*
valueB:Č*
dtype0*
_output_shapes
:
f
lstm/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
©
lstm/strided_slice_8StridedSlicelstm/ReadVariableOp_5lstm/strided_slice_8/stacklstm/strided_slice_8/stack_1lstm/strided_slice_8/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:d*
T0*
Index0

lstm/BiasAdd_1BiasAddlstm/MatMul_1lstm/strided_slice_8*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’d*
T0
\
lstm/ReadVariableOp_6ReadVariableOp	lstm/bias*
dtype0*
_output_shapes	
:
e
lstm/strided_slice_9/stackConst*
valueB:Č*
dtype0*
_output_shapes
:
g
lstm/strided_slice_9/stack_1Const*
_output_shapes
:*
valueB:¬*
dtype0
f
lstm/strided_slice_9/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
©
lstm/strided_slice_9StridedSlicelstm/ReadVariableOp_6lstm/strided_slice_9/stacklstm/strided_slice_9/stack_1lstm/strided_slice_9/stack_2*
_output_shapes
:d*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 

lstm/BiasAdd_2BiasAddlstm/MatMul_2lstm/strided_slice_9*'
_output_shapes
:’’’’’’’’’d*
T0*
data_formatNHWC
\
lstm/ReadVariableOp_7ReadVariableOp	lstm/bias*
_output_shapes	
:*
dtype0
f
lstm/strided_slice_10/stackConst*
valueB:¬*
dtype0*
_output_shapes
:
g
lstm/strided_slice_10/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
g
lstm/strided_slice_10/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
­
lstm/strided_slice_10StridedSlicelstm/ReadVariableOp_7lstm/strided_slice_10/stacklstm/strided_slice_10/stack_1lstm/strided_slice_10/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
:d

lstm/BiasAdd_3BiasAddlstm/MatMul_3lstm/strided_slice_10*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’d
l
lstm/ReadVariableOp_8ReadVariableOplstm/recurrent_kernel*
dtype0*
_output_shapes
:	d
l
lstm/strided_slice_11/stackConst*
valueB"        *
dtype0*
_output_shapes
:
n
lstm/strided_slice_11/stack_1Const*
valueB"    d   *
dtype0*
_output_shapes
:
n
lstm/strided_slice_11/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
±
lstm/strided_slice_11StridedSlicelstm/ReadVariableOp_8lstm/strided_slice_11/stacklstm/strided_slice_11/stack_1lstm/strided_slice_11/stack_2*
new_axis_mask *
end_mask*
_output_shapes

:dd*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask

lstm/MatMul_4MatMul
lstm/zeroslstm/strided_slice_11*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( *
T0
^
lstm/addAddlstm/BiasAddlstm/MatMul_4*'
_output_shapes
:’’’’’’’’’d*
T0
O

lstm/mul/xConst*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 
W
lstm/mulMul
lstm/mul/xlstm/add*
T0*'
_output_shapes
:’’’’’’’’’d
Q
lstm/add_1/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
[

lstm/add_1Addlstm/mullstm/add_1/y*
T0*'
_output_shapes
:’’’’’’’’’d
O

lstm/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Q
lstm/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
q
lstm/clip_by_value/MinimumMinimum
lstm/add_1lstm/Const_1*'
_output_shapes
:’’’’’’’’’d*
T0
w
lstm/clip_by_valueMaximumlstm/clip_by_value/Minimum
lstm/Const*'
_output_shapes
:’’’’’’’’’d*
T0
l
lstm/ReadVariableOp_9ReadVariableOplstm/recurrent_kernel*
dtype0*
_output_shapes
:	d
l
lstm/strided_slice_12/stackConst*
valueB"    d   *
dtype0*
_output_shapes
:
n
lstm/strided_slice_12/stack_1Const*
valueB"    Č   *
dtype0*
_output_shapes
:
n
lstm/strided_slice_12/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
±
lstm/strided_slice_12StridedSlicelstm/ReadVariableOp_9lstm/strided_slice_12/stacklstm/strided_slice_12/stack_1lstm/strided_slice_12/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:dd*
Index0*
T0

lstm/MatMul_5MatMul
lstm/zeroslstm/strided_slice_12*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
b

lstm/add_2Addlstm/BiasAdd_1lstm/MatMul_5*
T0*'
_output_shapes
:’’’’’’’’’d
Q
lstm/mul_1/xConst*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 
]

lstm/mul_1Mullstm/mul_1/x
lstm/add_2*
T0*'
_output_shapes
:’’’’’’’’’d
Q
lstm/add_3/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
]

lstm/add_3Add
lstm/mul_1lstm/add_3/y*'
_output_shapes
:’’’’’’’’’d*
T0
Q
lstm/Const_2Const*
_output_shapes
: *
valueB
 *    *
dtype0
Q
lstm/Const_3Const*
_output_shapes
: *
valueB
 *  ?*
dtype0
s
lstm/clip_by_value_1/MinimumMinimum
lstm/add_3lstm/Const_3*'
_output_shapes
:’’’’’’’’’d*
T0
}
lstm/clip_by_value_1Maximumlstm/clip_by_value_1/Minimumlstm/Const_2*'
_output_shapes
:’’’’’’’’’d*
T0
g

lstm/mul_2Mullstm/clip_by_value_1lstm/zeros_1*'
_output_shapes
:’’’’’’’’’d*
T0
m
lstm/ReadVariableOp_10ReadVariableOplstm/recurrent_kernel*
dtype0*
_output_shapes
:	d
l
lstm/strided_slice_13/stackConst*
_output_shapes
:*
valueB"    Č   *
dtype0
n
lstm/strided_slice_13/stack_1Const*
valueB"    ,  *
dtype0*
_output_shapes
:
n
lstm/strided_slice_13/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
²
lstm/strided_slice_13StridedSlicelstm/ReadVariableOp_10lstm/strided_slice_13/stacklstm/strided_slice_13/stack_1lstm/strided_slice_13/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:dd*
Index0*
T0

lstm/MatMul_6MatMul
lstm/zeroslstm/strided_slice_13*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
b

lstm/add_4Addlstm/BiasAdd_2lstm/MatMul_6*
T0*'
_output_shapes
:’’’’’’’’’d
O
	lstm/TanhTanh
lstm/add_4*'
_output_shapes
:’’’’’’’’’d*
T0
b

lstm/mul_3Mullstm/clip_by_value	lstm/Tanh*'
_output_shapes
:’’’’’’’’’d*
T0
[

lstm/add_5Add
lstm/mul_2
lstm/mul_3*'
_output_shapes
:’’’’’’’’’d*
T0
m
lstm/ReadVariableOp_11ReadVariableOplstm/recurrent_kernel*
dtype0*
_output_shapes
:	d
l
lstm/strided_slice_14/stackConst*
valueB"    ,  *
dtype0*
_output_shapes
:
n
lstm/strided_slice_14/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0
n
lstm/strided_slice_14/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
²
lstm/strided_slice_14StridedSlicelstm/ReadVariableOp_11lstm/strided_slice_14/stacklstm/strided_slice_14/stack_1lstm/strided_slice_14/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:dd*
T0*
Index0

lstm/MatMul_7MatMul
lstm/zeroslstm/strided_slice_14*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
b

lstm/add_6Addlstm/BiasAdd_3lstm/MatMul_7*
T0*'
_output_shapes
:’’’’’’’’’d
Q
lstm/mul_4/xConst*
_output_shapes
: *
valueB
 *ĶĢL>*
dtype0
]

lstm/mul_4Mullstm/mul_4/x
lstm/add_6*
T0*'
_output_shapes
:’’’’’’’’’d
Q
lstm/add_7/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
]

lstm/add_7Add
lstm/mul_4lstm/add_7/y*
T0*'
_output_shapes
:’’’’’’’’’d
Q
lstm/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Q
lstm/Const_5Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
s
lstm/clip_by_value_2/MinimumMinimum
lstm/add_7lstm/Const_5*
T0*'
_output_shapes
:’’’’’’’’’d
}
lstm/clip_by_value_2Maximumlstm/clip_by_value_2/Minimumlstm/Const_4*'
_output_shapes
:’’’’’’’’’d*
T0
Q
lstm/Tanh_1Tanh
lstm/add_5*'
_output_shapes
:’’’’’’’’’d*
T0
f

lstm/mul_5Mullstm/clip_by_value_2lstm/Tanh_1*
T0*'
_output_shapes
:’’’’’’’’’d
ē
lstm/TensorArrayTensorArrayV3lstm/strided_slice_1*
identical_element_shapes(* 
tensor_array_name	output_ta*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(
č
lstm/TensorArray_1TensorArrayV3lstm/strided_slice_1*
tensor_array_name
input_ta*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
k
lstm/TensorArrayUnstack/ShapeShapelstm/transpose*
T0*
out_type0*
_output_shapes
:
u
+lstm/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
-lstm/TensorArrayUnstack/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
w
-lstm/TensorArrayUnstack/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
ń
%lstm/TensorArrayUnstack/strided_sliceStridedSlicelstm/TensorArrayUnstack/Shape+lstm/TensorArrayUnstack/strided_slice/stack-lstm/TensorArrayUnstack/strided_slice/stack_1-lstm/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
e
#lstm/TensorArrayUnstack/range/startConst*
_output_shapes
: *
value	B : *
dtype0
e
#lstm/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Č
lstm/TensorArrayUnstack/rangeRange#lstm/TensorArrayUnstack/range/start%lstm/TensorArrayUnstack/strided_slice#lstm/TensorArrayUnstack/range/delta*#
_output_shapes
:’’’’’’’’’*

Tidx0
ō
?lstm/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3lstm/TensorArray_1lstm/TensorArrayUnstack/rangelstm/transposelstm/TensorArray_1:1*!
_class
loc:@lstm/transpose*
_output_shapes
: *
T0
K
	lstm/timeConst*
_output_shapes
: *
value	B : *
dtype0

lstm/while/EnterEnter	lstm/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *(

frame_namelstm/while/while_context
Ø
lstm/while/Enter_1Enterlstm/TensorArray:1*
_output_shapes
: *(

frame_namelstm/while/while_context*
T0*
is_constant( *
parallel_iterations 
±
lstm/while/Enter_2Enter
lstm/zeros*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:’’’’’’’’’d*(

frame_namelstm/while/while_context
³
lstm/while/Enter_3Enterlstm/zeros_1*'
_output_shapes
:’’’’’’’’’d*(

frame_namelstm/while/while_context*
T0*
is_constant( *
parallel_iterations 
q
lstm/while/MergeMergelstm/while/Enterlstm/while/NextIteration*
_output_shapes
: : *
T0*
N
w
lstm/while/Merge_1Mergelstm/while/Enter_1lstm/while/NextIteration_1*
T0*
N*
_output_shapes
: : 

lstm/while/Merge_2Mergelstm/while/Enter_2lstm/while/NextIteration_2*)
_output_shapes
:’’’’’’’’’d: *
T0*
N

lstm/while/Merge_3Mergelstm/while/Enter_3lstm/while/NextIteration_3*
T0*
N*)
_output_shapes
:’’’’’’’’’d: 
a
lstm/while/LessLesslstm/while/Mergelstm/while/Less/Enter*
_output_shapes
: *
T0
­
lstm/while/Less/EnterEnterlstm/strided_slice_1*
is_constant(*
parallel_iterations *
_output_shapes
: *(

frame_namelstm/while/while_context*
T0
H
lstm/while/LoopCondLoopCondlstm/while/Less*
_output_shapes
: 

lstm/while/SwitchSwitchlstm/while/Mergelstm/while/LoopCond*
_output_shapes
: : *
T0*#
_class
loc:@lstm/while/Merge

lstm/while/Switch_1Switchlstm/while/Merge_1lstm/while/LoopCond*%
_class
loc:@lstm/while/Merge_1*
_output_shapes
: : *
T0
²
lstm/while/Switch_2Switchlstm/while/Merge_2lstm/while/LoopCond*:
_output_shapes(
&:’’’’’’’’’d:’’’’’’’’’d*
T0*%
_class
loc:@lstm/while/Merge_2
²
lstm/while/Switch_3Switchlstm/while/Merge_3lstm/while/LoopCond*%
_class
loc:@lstm/while/Merge_3*:
_output_shapes(
&:’’’’’’’’’d:’’’’’’’’’d*
T0
U
lstm/while/IdentityIdentitylstm/while/Switch:1*
T0*
_output_shapes
: 
Y
lstm/while/Identity_1Identitylstm/while/Switch_1:1*
T0*
_output_shapes
: 
j
lstm/while/Identity_2Identitylstm/while/Switch_2:1*
T0*'
_output_shapes
:’’’’’’’’’d
j
lstm/while/Identity_3Identitylstm/while/Switch_3:1*'
_output_shapes
:’’’’’’’’’d*
T0
Ę
lstm/while/TensorArrayReadV3TensorArrayReadV3"lstm/while/TensorArrayReadV3/Enterlstm/while/Identity$lstm/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:’’’’’’’’’d
¼
"lstm/while/TensorArrayReadV3/EnterEnterlstm/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*(

frame_namelstm/while/while_context
ē
$lstm/while/TensorArrayReadV3/Enter_1Enter?lstm/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: *(

frame_namelstm/while/while_context*
T0*
is_constant(*
parallel_iterations 

lstm/while/ReadVariableOpReadVariableOplstm/while/ReadVariableOp/Enter^lstm/while/Identity*
dtype0*
_output_shapes
:	d
®
lstm/while/ReadVariableOp/EnterEnterlstm/kernel*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *(

frame_namelstm/while/while_context

lstm/while/strided_slice/stackConst^lstm/while/Identity*
valueB"        *
dtype0*
_output_shapes
:

 lstm/while/strided_slice/stack_1Const^lstm/while/Identity*
valueB"    d   *
dtype0*
_output_shapes
:

 lstm/while/strided_slice/stack_2Const^lstm/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Į
lstm/while/strided_sliceStridedSlicelstm/while/ReadVariableOplstm/while/strided_slice/stack lstm/while/strided_slice/stack_1 lstm/while/strided_slice/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:dd*
Index0*
T0
«
lstm/while/MatMulMatMullstm/while/TensorArrayReadV3lstm/while/strided_slice*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 

lstm/while/ReadVariableOp_1ReadVariableOplstm/while/ReadVariableOp/Enter^lstm/while/Identity*
dtype0*
_output_shapes
:	d

 lstm/while/strided_slice_1/stackConst^lstm/while/Identity*
valueB"    d   *
dtype0*
_output_shapes
:

"lstm/while/strided_slice_1/stack_1Const^lstm/while/Identity*
valueB"    Č   *
dtype0*
_output_shapes
:

"lstm/while/strided_slice_1/stack_2Const^lstm/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Ė
lstm/while/strided_slice_1StridedSlicelstm/while/ReadVariableOp_1 lstm/while/strided_slice_1/stack"lstm/while/strided_slice_1/stack_1"lstm/while/strided_slice_1/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:dd*
T0*
Index0
Æ
lstm/while/MatMul_1MatMullstm/while/TensorArrayReadV3lstm/while/strided_slice_1*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 

lstm/while/ReadVariableOp_2ReadVariableOplstm/while/ReadVariableOp/Enter^lstm/while/Identity*
dtype0*
_output_shapes
:	d

 lstm/while/strided_slice_2/stackConst^lstm/while/Identity*
valueB"    Č   *
dtype0*
_output_shapes
:

"lstm/while/strided_slice_2/stack_1Const^lstm/while/Identity*
valueB"    ,  *
dtype0*
_output_shapes
:

"lstm/while/strided_slice_2/stack_2Const^lstm/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Ė
lstm/while/strided_slice_2StridedSlicelstm/while/ReadVariableOp_2 lstm/while/strided_slice_2/stack"lstm/while/strided_slice_2/stack_1"lstm/while/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:dd
Æ
lstm/while/MatMul_2MatMullstm/while/TensorArrayReadV3lstm/while/strided_slice_2*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( *
T0

lstm/while/ReadVariableOp_3ReadVariableOplstm/while/ReadVariableOp/Enter^lstm/while/Identity*
dtype0*
_output_shapes
:	d

 lstm/while/strided_slice_3/stackConst^lstm/while/Identity*
valueB"    ,  *
dtype0*
_output_shapes
:

"lstm/while/strided_slice_3/stack_1Const^lstm/while/Identity*
valueB"        *
dtype0*
_output_shapes
:

"lstm/while/strided_slice_3/stack_2Const^lstm/while/Identity*
_output_shapes
:*
valueB"      *
dtype0
Ė
lstm/while/strided_slice_3StridedSlicelstm/while/ReadVariableOp_3 lstm/while/strided_slice_3/stack"lstm/while/strided_slice_3/stack_1"lstm/while/strided_slice_3/stack_2*
_output_shapes

:dd*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
Æ
lstm/while/MatMul_3MatMullstm/while/TensorArrayReadV3lstm/while/strided_slice_3*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( *
T0

lstm/while/ReadVariableOp_4ReadVariableOp!lstm/while/ReadVariableOp_4/Enter^lstm/while/Identity*
dtype0*
_output_shapes	
:
®
!lstm/while/ReadVariableOp_4/EnterEnter	lstm/bias*
is_constant(*
parallel_iterations *
_output_shapes
: *(

frame_namelstm/while/while_context*
T0

 lstm/while/strided_slice_4/stackConst^lstm/while/Identity*
valueB: *
dtype0*
_output_shapes
:

"lstm/while/strided_slice_4/stack_1Const^lstm/while/Identity*
valueB:d*
dtype0*
_output_shapes
:

"lstm/while/strided_slice_4/stack_2Const^lstm/while/Identity*
_output_shapes
:*
valueB:*
dtype0
Ē
lstm/while/strided_slice_4StridedSlicelstm/while/ReadVariableOp_4 lstm/while/strided_slice_4/stack"lstm/while/strided_slice_4/stack_1"lstm/while/strided_slice_4/stack_2*
_output_shapes
:d*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask 

lstm/while/BiasAddBiasAddlstm/while/MatMullstm/while/strided_slice_4*'
_output_shapes
:’’’’’’’’’d*
T0*
data_formatNHWC

lstm/while/ReadVariableOp_5ReadVariableOp!lstm/while/ReadVariableOp_4/Enter^lstm/while/Identity*
dtype0*
_output_shapes	
:

 lstm/while/strided_slice_5/stackConst^lstm/while/Identity*
valueB:d*
dtype0*
_output_shapes
:

"lstm/while/strided_slice_5/stack_1Const^lstm/while/Identity*
valueB:Č*
dtype0*
_output_shapes
:

"lstm/while/strided_slice_5/stack_2Const^lstm/while/Identity*
valueB:*
dtype0*
_output_shapes
:
Ē
lstm/while/strided_slice_5StridedSlicelstm/while/ReadVariableOp_5 lstm/while/strided_slice_5/stack"lstm/while/strided_slice_5/stack_1"lstm/while/strided_slice_5/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:d*
Index0*
T0

lstm/while/BiasAdd_1BiasAddlstm/while/MatMul_1lstm/while/strided_slice_5*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’d

lstm/while/ReadVariableOp_6ReadVariableOp!lstm/while/ReadVariableOp_4/Enter^lstm/while/Identity*
dtype0*
_output_shapes	
:

 lstm/while/strided_slice_6/stackConst^lstm/while/Identity*
valueB:Č*
dtype0*
_output_shapes
:

"lstm/while/strided_slice_6/stack_1Const^lstm/while/Identity*
valueB:¬*
dtype0*
_output_shapes
:

"lstm/while/strided_slice_6/stack_2Const^lstm/while/Identity*
valueB:*
dtype0*
_output_shapes
:
Ē
lstm/while/strided_slice_6StridedSlicelstm/while/ReadVariableOp_6 lstm/while/strided_slice_6/stack"lstm/while/strided_slice_6/stack_1"lstm/while/strided_slice_6/stack_2*
_output_shapes
:d*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 

lstm/while/BiasAdd_2BiasAddlstm/while/MatMul_2lstm/while/strided_slice_6*'
_output_shapes
:’’’’’’’’’d*
T0*
data_formatNHWC

lstm/while/ReadVariableOp_7ReadVariableOp!lstm/while/ReadVariableOp_4/Enter^lstm/while/Identity*
dtype0*
_output_shapes	
:

 lstm/while/strided_slice_7/stackConst^lstm/while/Identity*
valueB:¬*
dtype0*
_output_shapes
:

"lstm/while/strided_slice_7/stack_1Const^lstm/while/Identity*
valueB: *
dtype0*
_output_shapes
:

"lstm/while/strided_slice_7/stack_2Const^lstm/while/Identity*
valueB:*
dtype0*
_output_shapes
:
Ē
lstm/while/strided_slice_7StridedSlicelstm/while/ReadVariableOp_7 lstm/while/strided_slice_7/stack"lstm/while/strided_slice_7/stack_1"lstm/while/strided_slice_7/stack_2*
_output_shapes
:d*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask

lstm/while/BiasAdd_3BiasAddlstm/while/MatMul_3lstm/while/strided_slice_7*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’d*
T0

lstm/while/ReadVariableOp_8ReadVariableOp!lstm/while/ReadVariableOp_8/Enter^lstm/while/Identity*
dtype0*
_output_shapes
:	d
ŗ
!lstm/while/ReadVariableOp_8/EnterEnterlstm/recurrent_kernel*
_output_shapes
: *(

frame_namelstm/while/while_context*
T0*
is_constant(*
parallel_iterations 

 lstm/while/strided_slice_8/stackConst^lstm/while/Identity*
valueB"        *
dtype0*
_output_shapes
:

"lstm/while/strided_slice_8/stack_1Const^lstm/while/Identity*
valueB"    d   *
dtype0*
_output_shapes
:

"lstm/while/strided_slice_8/stack_2Const^lstm/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Ė
lstm/while/strided_slice_8StridedSlicelstm/while/ReadVariableOp_8 lstm/while/strided_slice_8/stack"lstm/while/strided_slice_8/stack_1"lstm/while/strided_slice_8/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:dd
Ø
lstm/while/MatMul_4MatMullstm/while/Identity_2lstm/while/strided_slice_8*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
p
lstm/while/addAddlstm/while/BiasAddlstm/while/MatMul_4*'
_output_shapes
:’’’’’’’’’d*
T0
k
lstm/while/mul/xConst^lstm/while/Identity*
_output_shapes
: *
valueB
 *ĶĢL>*
dtype0
i
lstm/while/mulMullstm/while/mul/xlstm/while/add*'
_output_shapes
:’’’’’’’’’d*
T0
m
lstm/while/add_1/yConst^lstm/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
m
lstm/while/add_1Addlstm/while/mullstm/while/add_1/y*
T0*'
_output_shapes
:’’’’’’’’’d
k
lstm/while/ConstConst^lstm/while/Identity*
_output_shapes
: *
valueB
 *    *
dtype0
m
lstm/while/Const_1Const^lstm/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

 lstm/while/clip_by_value/MinimumMinimumlstm/while/add_1lstm/while/Const_1*'
_output_shapes
:’’’’’’’’’d*
T0

lstm/while/clip_by_valueMaximum lstm/while/clip_by_value/Minimumlstm/while/Const*'
_output_shapes
:’’’’’’’’’d*
T0

lstm/while/ReadVariableOp_9ReadVariableOp!lstm/while/ReadVariableOp_8/Enter^lstm/while/Identity*
dtype0*
_output_shapes
:	d

 lstm/while/strided_slice_9/stackConst^lstm/while/Identity*
valueB"    d   *
dtype0*
_output_shapes
:

"lstm/while/strided_slice_9/stack_1Const^lstm/while/Identity*
valueB"    Č   *
dtype0*
_output_shapes
:

"lstm/while/strided_slice_9/stack_2Const^lstm/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Ė
lstm/while/strided_slice_9StridedSlicelstm/while/ReadVariableOp_9 lstm/while/strided_slice_9/stack"lstm/while/strided_slice_9/stack_1"lstm/while/strided_slice_9/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:dd
Ø
lstm/while/MatMul_5MatMullstm/while/Identity_2lstm/while/strided_slice_9*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( *
T0
t
lstm/while/add_2Addlstm/while/BiasAdd_1lstm/while/MatMul_5*
T0*'
_output_shapes
:’’’’’’’’’d
m
lstm/while/mul_1/xConst^lstm/while/Identity*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 
o
lstm/while/mul_1Mullstm/while/mul_1/xlstm/while/add_2*
T0*'
_output_shapes
:’’’’’’’’’d
m
lstm/while/add_3/yConst^lstm/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
o
lstm/while/add_3Addlstm/while/mul_1lstm/while/add_3/y*'
_output_shapes
:’’’’’’’’’d*
T0
m
lstm/while/Const_2Const^lstm/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
m
lstm/while/Const_3Const^lstm/while/Identity*
_output_shapes
: *
valueB
 *  ?*
dtype0

"lstm/while/clip_by_value_1/MinimumMinimumlstm/while/add_3lstm/while/Const_3*
T0*'
_output_shapes
:’’’’’’’’’d

lstm/while/clip_by_value_1Maximum"lstm/while/clip_by_value_1/Minimumlstm/while/Const_2*'
_output_shapes
:’’’’’’’’’d*
T0
|
lstm/while/mul_2Mullstm/while/clip_by_value_1lstm/while/Identity_3*
T0*'
_output_shapes
:’’’’’’’’’d

lstm/while/ReadVariableOp_10ReadVariableOp!lstm/while/ReadVariableOp_8/Enter^lstm/while/Identity*
_output_shapes
:	d*
dtype0

!lstm/while/strided_slice_10/stackConst^lstm/while/Identity*
valueB"    Č   *
dtype0*
_output_shapes
:

#lstm/while/strided_slice_10/stack_1Const^lstm/while/Identity*
_output_shapes
:*
valueB"    ,  *
dtype0

#lstm/while/strided_slice_10/stack_2Const^lstm/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Š
lstm/while/strided_slice_10StridedSlicelstm/while/ReadVariableOp_10!lstm/while/strided_slice_10/stack#lstm/while/strided_slice_10/stack_1#lstm/while/strided_slice_10/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:dd*
Index0*
T0
©
lstm/while/MatMul_6MatMullstm/while/Identity_2lstm/while/strided_slice_10*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
t
lstm/while/add_4Addlstm/while/BiasAdd_2lstm/while/MatMul_6*'
_output_shapes
:’’’’’’’’’d*
T0
[
lstm/while/TanhTanhlstm/while/add_4*'
_output_shapes
:’’’’’’’’’d*
T0
t
lstm/while/mul_3Mullstm/while/clip_by_valuelstm/while/Tanh*'
_output_shapes
:’’’’’’’’’d*
T0
m
lstm/while/add_5Addlstm/while/mul_2lstm/while/mul_3*
T0*'
_output_shapes
:’’’’’’’’’d

lstm/while/ReadVariableOp_11ReadVariableOp!lstm/while/ReadVariableOp_8/Enter^lstm/while/Identity*
dtype0*
_output_shapes
:	d

!lstm/while/strided_slice_11/stackConst^lstm/while/Identity*
_output_shapes
:*
valueB"    ,  *
dtype0

#lstm/while/strided_slice_11/stack_1Const^lstm/while/Identity*
valueB"        *
dtype0*
_output_shapes
:

#lstm/while/strided_slice_11/stack_2Const^lstm/while/Identity*
_output_shapes
:*
valueB"      *
dtype0
Š
lstm/while/strided_slice_11StridedSlicelstm/while/ReadVariableOp_11!lstm/while/strided_slice_11/stack#lstm/while/strided_slice_11/stack_1#lstm/while/strided_slice_11/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:dd*
T0*
Index0*
shrink_axis_mask 
©
lstm/while/MatMul_7MatMullstm/while/Identity_2lstm/while/strided_slice_11*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( *
T0
t
lstm/while/add_6Addlstm/while/BiasAdd_3lstm/while/MatMul_7*
T0*'
_output_shapes
:’’’’’’’’’d
m
lstm/while/mul_4/xConst^lstm/while/Identity*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 
o
lstm/while/mul_4Mullstm/while/mul_4/xlstm/while/add_6*
T0*'
_output_shapes
:’’’’’’’’’d
m
lstm/while/add_7/yConst^lstm/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
o
lstm/while/add_7Addlstm/while/mul_4lstm/while/add_7/y*'
_output_shapes
:’’’’’’’’’d*
T0
m
lstm/while/Const_4Const^lstm/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
m
lstm/while/Const_5Const^lstm/while/Identity*
_output_shapes
: *
valueB
 *  ?*
dtype0

"lstm/while/clip_by_value_2/MinimumMinimumlstm/while/add_7lstm/while/Const_5*
T0*'
_output_shapes
:’’’’’’’’’d

lstm/while/clip_by_value_2Maximum"lstm/while/clip_by_value_2/Minimumlstm/while/Const_4*'
_output_shapes
:’’’’’’’’’d*
T0
]
lstm/while/Tanh_1Tanhlstm/while/add_5*
T0*'
_output_shapes
:’’’’’’’’’d
x
lstm/while/mul_5Mullstm/while/clip_by_value_2lstm/while/Tanh_1*
T0*'
_output_shapes
:’’’’’’’’’d
ž
.lstm/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV34lstm/while/TensorArrayWrite/TensorArrayWriteV3/Enterlstm/while/Identitylstm/while/mul_5lstm/while/Identity_1*
T0*#
_class
loc:@lstm/while/mul_5*
_output_shapes
: 
ń
4lstm/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlstm/TensorArray*
T0*#
_class
loc:@lstm/while/mul_5*
parallel_iterations *
is_constant(*
_output_shapes
:*(

frame_namelstm/while/while_context
j
lstm/while/add_8/yConst^lstm/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
a
lstm/while/add_8Addlstm/while/Identitylstm/while/add_8/y*
_output_shapes
: *
T0
\
lstm/while/NextIterationNextIterationlstm/while/add_8*
T0*
_output_shapes
: 
|
lstm/while/NextIteration_1NextIteration.lstm/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
o
lstm/while/NextIteration_2NextIterationlstm/while/mul_5*'
_output_shapes
:’’’’’’’’’d*
T0
o
lstm/while/NextIteration_3NextIterationlstm/while/add_5*'
_output_shapes
:’’’’’’’’’d*
T0
K
lstm/while/ExitExitlstm/while/Switch*
T0*
_output_shapes
: 
O
lstm/while/Exit_1Exitlstm/while/Switch_1*
T0*
_output_shapes
: 
`
lstm/while/Exit_2Exitlstm/while/Switch_2*'
_output_shapes
:’’’’’’’’’d*
T0
`
lstm/while/Exit_3Exitlstm/while/Switch_3*'
_output_shapes
:’’’’’’’’’d*
T0

'lstm/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lstm/TensorArraylstm/while/Exit_1*#
_class
loc:@lstm/TensorArray*
_output_shapes
: 

!lstm/TensorArrayStack/range/startConst*#
_class
loc:@lstm/TensorArray*
value	B : *
dtype0*
_output_shapes
: 

!lstm/TensorArrayStack/range/deltaConst*
_output_shapes
: *#
_class
loc:@lstm/TensorArray*
value	B :*
dtype0
é
lstm/TensorArrayStack/rangeRange!lstm/TensorArrayStack/range/start'lstm/TensorArrayStack/TensorArraySizeV3!lstm/TensorArrayStack/range/delta*

Tidx0*#
_class
loc:@lstm/TensorArray*#
_output_shapes
:’’’’’’’’’

)lstm/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lstm/TensorArraylstm/TensorArrayStack/rangelstm/while/Exit_1*#
_class
loc:@lstm/TensorArray*
dtype0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d*$
element_shape:’’’’’’’’’d
L

lstm/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
M
lstm/subSublstm/while/Exit
lstm/sub/y*
T0*
_output_shapes
: 

lstm/TensorArrayReadV3TensorArrayReadV3lstm/TensorArraylstm/sublstm/while/Exit_1*
dtype0*'
_output_shapes
:’’’’’’’’’d
j
lstm/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
«
lstm/transpose_1	Transpose)lstm/TensorArrayStack/TensorArrayGatherV3lstm/transpose_1/perm*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d*
Tperm0
”
.lstm_1/kernel/Initializer/random_uniform/shapeConst* 
_class
loc:@lstm_1/kernel*
valueB"d     *
dtype0*
_output_shapes
:

,lstm_1/kernel/Initializer/random_uniform/minConst* 
_class
loc:@lstm_1/kernel*
valueB
 *ßXą½*
dtype0*
_output_shapes
: 

,lstm_1/kernel/Initializer/random_uniform/maxConst* 
_class
loc:@lstm_1/kernel*
valueB
 *ßXą=*
dtype0*
_output_shapes
: 
é
6lstm_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform.lstm_1/kernel/Initializer/random_uniform/shape*
_output_shapes
:	d*

seed *
T0* 
_class
loc:@lstm_1/kernel*
seed2 *
dtype0
Ņ
,lstm_1/kernel/Initializer/random_uniform/subSub,lstm_1/kernel/Initializer/random_uniform/max,lstm_1/kernel/Initializer/random_uniform/min* 
_class
loc:@lstm_1/kernel*
_output_shapes
: *
T0
å
,lstm_1/kernel/Initializer/random_uniform/mulMul6lstm_1/kernel/Initializer/random_uniform/RandomUniform,lstm_1/kernel/Initializer/random_uniform/sub*
_output_shapes
:	d*
T0* 
_class
loc:@lstm_1/kernel
×
(lstm_1/kernel/Initializer/random_uniformAdd,lstm_1/kernel/Initializer/random_uniform/mul,lstm_1/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@lstm_1/kernel*
_output_shapes
:	d
Ŗ
lstm_1/kernelVarHandleOp*
	container *
shape:	d*
dtype0*
_output_shapes
: *
shared_namelstm_1/kernel* 
_class
loc:@lstm_1/kernel
k
.lstm_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOplstm_1/kernel*
_output_shapes
: 

lstm_1/kernel/AssignAssignVariableOplstm_1/kernel(lstm_1/kernel/Initializer/random_uniform* 
_class
loc:@lstm_1/kernel*
dtype0

!lstm_1/kernel/Read/ReadVariableOpReadVariableOplstm_1/kernel* 
_class
loc:@lstm_1/kernel*
dtype0*
_output_shapes
:	d
“
7lstm_1/recurrent_kernel/Initializer/random_normal/shapeConst**
_class 
loc:@lstm_1/recurrent_kernel*
valueB"  d   *
dtype0*
_output_shapes
:
§
6lstm_1/recurrent_kernel/Initializer/random_normal/meanConst**
_class 
loc:@lstm_1/recurrent_kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
©
8lstm_1/recurrent_kernel/Initializer/random_normal/stddevConst**
_class 
loc:@lstm_1/recurrent_kernel*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Flstm_1/recurrent_kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal7lstm_1/recurrent_kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes
:	d*

seed *
T0**
_class 
loc:@lstm_1/recurrent_kernel*
seed2 

5lstm_1/recurrent_kernel/Initializer/random_normal/mulMulFlstm_1/recurrent_kernel/Initializer/random_normal/RandomStandardNormal8lstm_1/recurrent_kernel/Initializer/random_normal/stddev**
_class 
loc:@lstm_1/recurrent_kernel*
_output_shapes
:	d*
T0
ż
1lstm_1/recurrent_kernel/Initializer/random_normalAdd5lstm_1/recurrent_kernel/Initializer/random_normal/mul6lstm_1/recurrent_kernel/Initializer/random_normal/mean*
_output_shapes
:	d*
T0**
_class 
loc:@lstm_1/recurrent_kernel
Ō
&lstm_1/recurrent_kernel/Initializer/QrQr1lstm_1/recurrent_kernel/Initializer/random_normal*
full_matrices( *
T0**
_class 
loc:@lstm_1/recurrent_kernel*)
_output_shapes
:	d:dd
³
,lstm_1/recurrent_kernel/Initializer/DiagPartDiagPart(lstm_1/recurrent_kernel/Initializer/Qr:1*
T0**
_class 
loc:@lstm_1/recurrent_kernel*
_output_shapes
:d
Æ
(lstm_1/recurrent_kernel/Initializer/SignSign,lstm_1/recurrent_kernel/Initializer/DiagPart*
T0**
_class 
loc:@lstm_1/recurrent_kernel*
_output_shapes
:d
Ö
'lstm_1/recurrent_kernel/Initializer/mulMul&lstm_1/recurrent_kernel/Initializer/Qr(lstm_1/recurrent_kernel/Initializer/Sign*
T0**
_class 
loc:@lstm_1/recurrent_kernel*
_output_shapes
:	d
Ą
Clstm_1/recurrent_kernel/Initializer/matrix_transpose/transpose/permConst**
_class 
loc:@lstm_1/recurrent_kernel*
valueB"       *
dtype0*
_output_shapes
:

>lstm_1/recurrent_kernel/Initializer/matrix_transpose/transpose	Transpose'lstm_1/recurrent_kernel/Initializer/mulClstm_1/recurrent_kernel/Initializer/matrix_transpose/transpose/perm*
T0**
_class 
loc:@lstm_1/recurrent_kernel*
_output_shapes
:	d*
Tperm0
®
1lstm_1/recurrent_kernel/Initializer/Reshape/shapeConst*
_output_shapes
:**
_class 
loc:@lstm_1/recurrent_kernel*
valueB"d     *
dtype0

+lstm_1/recurrent_kernel/Initializer/ReshapeReshape>lstm_1/recurrent_kernel/Initializer/matrix_transpose/transpose1lstm_1/recurrent_kernel/Initializer/Reshape/shape**
_class 
loc:@lstm_1/recurrent_kernel*
Tshape0*
_output_shapes
:	d*
T0

+lstm_1/recurrent_kernel/Initializer/mul_1/xConst*
_output_shapes
: **
_class 
loc:@lstm_1/recurrent_kernel*
valueB
 *  ?*
dtype0
ą
)lstm_1/recurrent_kernel/Initializer/mul_1Mul+lstm_1/recurrent_kernel/Initializer/mul_1/x+lstm_1/recurrent_kernel/Initializer/Reshape*
T0**
_class 
loc:@lstm_1/recurrent_kernel*
_output_shapes
:	d
Č
lstm_1/recurrent_kernelVarHandleOp*
	container *
shape:	d*
dtype0*
_output_shapes
: *(
shared_namelstm_1/recurrent_kernel**
_class 
loc:@lstm_1/recurrent_kernel

8lstm_1/recurrent_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOplstm_1/recurrent_kernel*
_output_shapes
: 
Æ
lstm_1/recurrent_kernel/AssignAssignVariableOplstm_1/recurrent_kernel)lstm_1/recurrent_kernel/Initializer/mul_1**
_class 
loc:@lstm_1/recurrent_kernel*
dtype0
°
+lstm_1/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm_1/recurrent_kernel*
_output_shapes
:	d**
_class 
loc:@lstm_1/recurrent_kernel*
dtype0

lstm_1/bias/Initializer/zerosConst*
_class
loc:@lstm_1/bias*
valueBd*    *
dtype0*
_output_shapes
:d

lstm_1/bias/Initializer/onesConst*
_output_shapes
:d*
_class
loc:@lstm_1/bias*
valueBd*  ?*
dtype0

lstm_1/bias/Initializer/zeros_1Const*
_class
loc:@lstm_1/bias*
valueBČ*    *
dtype0*
_output_shapes	
:Č

#lstm_1/bias/Initializer/concat/axisConst*
_class
loc:@lstm_1/bias*
value	B : *
dtype0*
_output_shapes
: 

lstm_1/bias/Initializer/concatConcatV2lstm_1/bias/Initializer/zeroslstm_1/bias/Initializer/oneslstm_1/bias/Initializer/zeros_1#lstm_1/bias/Initializer/concat/axis*
_output_shapes	
:*

Tidx0*
T0*
_class
loc:@lstm_1/bias*
N
 
lstm_1/biasVarHandleOp*
shape:*
dtype0*
_output_shapes
: *
shared_namelstm_1/bias*
_class
loc:@lstm_1/bias*
	container 
g
,lstm_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOplstm_1/bias*
_output_shapes
: 

lstm_1/bias/AssignAssignVariableOplstm_1/biaslstm_1/bias/Initializer/concat*
_class
loc:@lstm_1/bias*
dtype0

lstm_1/bias/Read/ReadVariableOpReadVariableOplstm_1/bias*
_output_shapes	
:*
_class
loc:@lstm_1/bias*
dtype0
^
lstm_1/ShapeShapedropout/cond/Merge*
_output_shapes
:*
T0*
out_type0
d
lstm_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
f
lstm_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
f
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0

lstm_1/strided_sliceStridedSlicelstm_1/Shapelstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
T
lstm_1/zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: 
b
lstm_1/zeros/mulMullstm_1/strided_slicelstm_1/zeros/mul/y*
_output_shapes
: *
T0
V
lstm_1/zeros/Less/yConst*
value
B :č*
dtype0*
_output_shapes
: 
a
lstm_1/zeros/LessLesslstm_1/zeros/mullstm_1/zeros/Less/y*
T0*
_output_shapes
: 
W
lstm_1/zeros/packed/1Const*
_output_shapes
: *
value	B :d*
dtype0

lstm_1/zeros/packedPacklstm_1/strided_slicelstm_1/zeros/packed/1*
T0*

axis *
N*
_output_shapes
:
W
lstm_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

lstm_1/zerosFilllstm_1/zeros/packedlstm_1/zeros/Const*'
_output_shapes
:’’’’’’’’’d*
T0*

index_type0
V
lstm_1/zeros_1/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: 
f
lstm_1/zeros_1/mulMullstm_1/strided_slicelstm_1/zeros_1/mul/y*
T0*
_output_shapes
: 
X
lstm_1/zeros_1/Less/yConst*
value
B :č*
dtype0*
_output_shapes
: 
g
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mullstm_1/zeros_1/Less/y*
T0*
_output_shapes
: 
Y
lstm_1/zeros_1/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: 

lstm_1/zeros_1/packedPacklstm_1/strided_slicelstm_1/zeros_1/packed/1*

axis *
N*
_output_shapes
:*
T0
Y
lstm_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

lstm_1/zeros_1Filllstm_1/zeros_1/packedlstm_1/zeros_1/Const*'
_output_shapes
:’’’’’’’’’d*
T0*

index_type0
j
lstm_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:

lstm_1/transpose	Transposedropout/cond/Mergelstm_1/transpose/perm*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d*
Tperm0*
T0
_
lstm_1/ReverseV2/axisConst*
valueB: *
dtype0*
_output_shapes
:

lstm_1/ReverseV2	ReverseV2lstm_1/transposelstm_1/ReverseV2/axis*

Tidx0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d
^
lstm_1/Shape_1Shapelstm_1/ReverseV2*
T0*
out_type0*
_output_shapes
:
f
lstm_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
h
lstm_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¦
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1lstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
f
lstm_1/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¹
lstm_1/strided_slice_2StridedSlicelstm_1/ReverseV2lstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2*
new_axis_mask *
end_mask *'
_output_shapes
:’’’’’’’’’d*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask 
d
lstm_1/ReadVariableOpReadVariableOplstm_1/kernel*
_output_shapes
:	d*
dtype0
m
lstm_1/strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_3/stack_1Const*
valueB"    d   *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
µ
lstm_1/strided_slice_3StridedSlicelstm_1/ReadVariableOplstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2*
new_axis_mask *
end_mask*
_output_shapes

:dd*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask 

lstm_1/MatMulMatMullstm_1/strided_slice_2lstm_1/strided_slice_3*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( *
T0
f
lstm_1/ReadVariableOp_1ReadVariableOplstm_1/kernel*
_output_shapes
:	d*
dtype0
m
lstm_1/strided_slice_4/stackConst*
valueB"    d   *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_4/stack_1Const*
valueB"    Č   *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
·
lstm_1/strided_slice_4StridedSlicelstm_1/ReadVariableOp_1lstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:dd*
T0*
Index0*
shrink_axis_mask 
”
lstm_1/MatMul_1MatMullstm_1/strided_slice_2lstm_1/strided_slice_4*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
f
lstm_1/ReadVariableOp_2ReadVariableOplstm_1/kernel*
dtype0*
_output_shapes
:	d
m
lstm_1/strided_slice_5/stackConst*
_output_shapes
:*
valueB"    Č   *
dtype0
o
lstm_1/strided_slice_5/stack_1Const*
valueB"    ,  *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
·
lstm_1/strided_slice_5StridedSlicelstm_1/ReadVariableOp_2lstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:dd*
T0*
Index0
”
lstm_1/MatMul_2MatMullstm_1/strided_slice_2lstm_1/strided_slice_5*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
f
lstm_1/ReadVariableOp_3ReadVariableOplstm_1/kernel*
dtype0*
_output_shapes
:	d
m
lstm_1/strided_slice_6/stackConst*
valueB"    ,  *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
·
lstm_1/strided_slice_6StridedSlicelstm_1/ReadVariableOp_3lstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:dd*
T0*
Index0
”
lstm_1/MatMul_3MatMullstm_1/strided_slice_2lstm_1/strided_slice_6*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
`
lstm_1/ReadVariableOp_4ReadVariableOplstm_1/bias*
dtype0*
_output_shapes	
:
f
lstm_1/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_7/stack_1Const*
dtype0*
_output_shapes
:*
valueB:d
h
lstm_1/strided_slice_7/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
³
lstm_1/strided_slice_7StridedSlicelstm_1/ReadVariableOp_4lstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:d

lstm_1/BiasAddBiasAddlstm_1/MatMullstm_1/strided_slice_7*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’d
`
lstm_1/ReadVariableOp_5ReadVariableOplstm_1/bias*
_output_shapes	
:*
dtype0
f
lstm_1/strided_slice_8/stackConst*
valueB:d*
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_8/stack_1Const*
valueB:Č*
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
³
lstm_1/strided_slice_8StridedSlicelstm_1/ReadVariableOp_5lstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:d*
Index0*
T0*
shrink_axis_mask 

lstm_1/BiasAdd_1BiasAddlstm_1/MatMul_1lstm_1/strided_slice_8*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’d
`
lstm_1/ReadVariableOp_6ReadVariableOplstm_1/bias*
_output_shapes	
:*
dtype0
g
lstm_1/strided_slice_9/stackConst*
valueB:Č*
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_9/stack_1Const*
_output_shapes
:*
valueB:¬*
dtype0
h
lstm_1/strided_slice_9/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
³
lstm_1/strided_slice_9StridedSlicelstm_1/ReadVariableOp_6lstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:d*
T0*
Index0

lstm_1/BiasAdd_2BiasAddlstm_1/MatMul_2lstm_1/strided_slice_9*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’d*
T0
`
lstm_1/ReadVariableOp_7ReadVariableOplstm_1/bias*
dtype0*
_output_shapes	
:
h
lstm_1/strided_slice_10/stackConst*
valueB:¬*
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
·
lstm_1/strided_slice_10StridedSlicelstm_1/ReadVariableOp_7lstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:d

lstm_1/BiasAdd_3BiasAddlstm_1/MatMul_3lstm_1/strided_slice_10*'
_output_shapes
:’’’’’’’’’d*
T0*
data_formatNHWC
p
lstm_1/ReadVariableOp_8ReadVariableOplstm_1/recurrent_kernel*
_output_shapes
:	d*
dtype0
n
lstm_1/strided_slice_11/stackConst*
valueB"        *
dtype0*
_output_shapes
:
p
lstm_1/strided_slice_11/stack_1Const*
_output_shapes
:*
valueB"    d   *
dtype0
p
lstm_1/strided_slice_11/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
»
lstm_1/strided_slice_11StridedSlicelstm_1/ReadVariableOp_8lstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:dd*
T0*
Index0

lstm_1/MatMul_4MatMullstm_1/zeroslstm_1/strided_slice_11*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( *
T0
d

lstm_1/addAddlstm_1/BiasAddlstm_1/MatMul_4*
T0*'
_output_shapes
:’’’’’’’’’d
Q
lstm_1/mul/xConst*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 
]

lstm_1/mulMullstm_1/mul/x
lstm_1/add*
T0*'
_output_shapes
:’’’’’’’’’d
S
lstm_1/add_1/yConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
a
lstm_1/add_1Add
lstm_1/mullstm_1/add_1/y*'
_output_shapes
:’’’’’’’’’d*
T0
Q
lstm_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
S
lstm_1/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
w
lstm_1/clip_by_value/MinimumMinimumlstm_1/add_1lstm_1/Const_1*
T0*'
_output_shapes
:’’’’’’’’’d
}
lstm_1/clip_by_valueMaximumlstm_1/clip_by_value/Minimumlstm_1/Const*
T0*'
_output_shapes
:’’’’’’’’’d
p
lstm_1/ReadVariableOp_9ReadVariableOplstm_1/recurrent_kernel*
_output_shapes
:	d*
dtype0
n
lstm_1/strided_slice_12/stackConst*
dtype0*
_output_shapes
:*
valueB"    d   
p
lstm_1/strided_slice_12/stack_1Const*
valueB"    Č   *
dtype0*
_output_shapes
:
p
lstm_1/strided_slice_12/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
»
lstm_1/strided_slice_12StridedSlicelstm_1/ReadVariableOp_9lstm_1/strided_slice_12/stacklstm_1/strided_slice_12/stack_1lstm_1/strided_slice_12/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:dd*
T0*
Index0

lstm_1/MatMul_5MatMullstm_1/zeroslstm_1/strided_slice_12*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
h
lstm_1/add_2Addlstm_1/BiasAdd_1lstm_1/MatMul_5*
T0*'
_output_shapes
:’’’’’’’’’d
S
lstm_1/mul_1/xConst*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 
c
lstm_1/mul_1Mullstm_1/mul_1/xlstm_1/add_2*'
_output_shapes
:’’’’’’’’’d*
T0
S
lstm_1/add_3/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
c
lstm_1/add_3Addlstm_1/mul_1lstm_1/add_3/y*'
_output_shapes
:’’’’’’’’’d*
T0
S
lstm_1/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
S
lstm_1/Const_3Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
y
lstm_1/clip_by_value_1/MinimumMinimumlstm_1/add_3lstm_1/Const_3*'
_output_shapes
:’’’’’’’’’d*
T0

lstm_1/clip_by_value_1Maximumlstm_1/clip_by_value_1/Minimumlstm_1/Const_2*'
_output_shapes
:’’’’’’’’’d*
T0
m
lstm_1/mul_2Mullstm_1/clip_by_value_1lstm_1/zeros_1*
T0*'
_output_shapes
:’’’’’’’’’d
q
lstm_1/ReadVariableOp_10ReadVariableOplstm_1/recurrent_kernel*
dtype0*
_output_shapes
:	d
n
lstm_1/strided_slice_13/stackConst*
valueB"    Č   *
dtype0*
_output_shapes
:
p
lstm_1/strided_slice_13/stack_1Const*
valueB"    ,  *
dtype0*
_output_shapes
:
p
lstm_1/strided_slice_13/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
¼
lstm_1/strided_slice_13StridedSlicelstm_1/ReadVariableOp_10lstm_1/strided_slice_13/stacklstm_1/strided_slice_13/stack_1lstm_1/strided_slice_13/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:dd*
T0*
Index0*
shrink_axis_mask 

lstm_1/MatMul_6MatMullstm_1/zeroslstm_1/strided_slice_13*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
h
lstm_1/add_4Addlstm_1/BiasAdd_2lstm_1/MatMul_6*'
_output_shapes
:’’’’’’’’’d*
T0
S
lstm_1/TanhTanhlstm_1/add_4*'
_output_shapes
:’’’’’’’’’d*
T0
h
lstm_1/mul_3Mullstm_1/clip_by_valuelstm_1/Tanh*
T0*'
_output_shapes
:’’’’’’’’’d
a
lstm_1/add_5Addlstm_1/mul_2lstm_1/mul_3*
T0*'
_output_shapes
:’’’’’’’’’d
q
lstm_1/ReadVariableOp_11ReadVariableOplstm_1/recurrent_kernel*
dtype0*
_output_shapes
:	d
n
lstm_1/strided_slice_14/stackConst*
valueB"    ,  *
dtype0*
_output_shapes
:
p
lstm_1/strided_slice_14/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
p
lstm_1/strided_slice_14/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
¼
lstm_1/strided_slice_14StridedSlicelstm_1/ReadVariableOp_11lstm_1/strided_slice_14/stacklstm_1/strided_slice_14/stack_1lstm_1/strided_slice_14/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:dd

lstm_1/MatMul_7MatMullstm_1/zeroslstm_1/strided_slice_14*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
h
lstm_1/add_6Addlstm_1/BiasAdd_3lstm_1/MatMul_7*'
_output_shapes
:’’’’’’’’’d*
T0
S
lstm_1/mul_4/xConst*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 
c
lstm_1/mul_4Mullstm_1/mul_4/xlstm_1/add_6*
T0*'
_output_shapes
:’’’’’’’’’d
S
lstm_1/add_7/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
c
lstm_1/add_7Addlstm_1/mul_4lstm_1/add_7/y*'
_output_shapes
:’’’’’’’’’d*
T0
S
lstm_1/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
S
lstm_1/Const_5Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
y
lstm_1/clip_by_value_2/MinimumMinimumlstm_1/add_7lstm_1/Const_5*'
_output_shapes
:’’’’’’’’’d*
T0

lstm_1/clip_by_value_2Maximumlstm_1/clip_by_value_2/Minimumlstm_1/Const_4*
T0*'
_output_shapes
:’’’’’’’’’d
U
lstm_1/Tanh_1Tanhlstm_1/add_5*'
_output_shapes
:’’’’’’’’’d*
T0
l
lstm_1/mul_5Mullstm_1/clip_by_value_2lstm_1/Tanh_1*
T0*'
_output_shapes
:’’’’’’’’’d
ė
lstm_1/TensorArrayTensorArrayV3lstm_1/strided_slice_1* 
tensor_array_name	output_ta*
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
ģ
lstm_1/TensorArray_1TensorArrayV3lstm_1/strided_slice_1*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*
tensor_array_name
input_ta*
dtype0*
_output_shapes

:: *
element_shape:
o
lstm_1/TensorArrayUnstack/ShapeShapelstm_1/ReverseV2*
_output_shapes
:*
T0*
out_type0
w
-lstm_1/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/lstm_1/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/lstm_1/TensorArrayUnstack/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
ū
'lstm_1/TensorArrayUnstack/strided_sliceStridedSlicelstm_1/TensorArrayUnstack/Shape-lstm_1/TensorArrayUnstack/strided_slice/stack/lstm_1/TensorArrayUnstack/strided_slice/stack_1/lstm_1/TensorArrayUnstack/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask 
g
%lstm_1/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%lstm_1/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Š
lstm_1/TensorArrayUnstack/rangeRange%lstm_1/TensorArrayUnstack/range/start'lstm_1/TensorArrayUnstack/strided_slice%lstm_1/TensorArrayUnstack/range/delta*#
_output_shapes
:’’’’’’’’’*

Tidx0

Alstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3lstm_1/TensorArray_1lstm_1/TensorArrayUnstack/rangelstm_1/ReverseV2lstm_1/TensorArray_1:1*
_output_shapes
: *
T0*#
_class
loc:@lstm_1/ReverseV2
M
lstm_1/timeConst*
value	B : *
dtype0*
_output_shapes
: 
£
lstm_1/while/EnterEnterlstm_1/time*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context*
T0
®
lstm_1/while/Enter_1Enterlstm_1/TensorArray:1*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context*
T0
·
lstm_1/while/Enter_2Enterlstm_1/zeros*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:’’’’’’’’’d**

frame_namelstm_1/while/while_context
¹
lstm_1/while/Enter_3Enterlstm_1/zeros_1*
is_constant( *
parallel_iterations *'
_output_shapes
:’’’’’’’’’d**

frame_namelstm_1/while/while_context*
T0
w
lstm_1/while/MergeMergelstm_1/while/Enterlstm_1/while/NextIteration*
T0*
N*
_output_shapes
: : 
}
lstm_1/while/Merge_1Mergelstm_1/while/Enter_1lstm_1/while/NextIteration_1*
T0*
N*
_output_shapes
: : 

lstm_1/while/Merge_2Mergelstm_1/while/Enter_2lstm_1/while/NextIteration_2*
T0*
N*)
_output_shapes
:’’’’’’’’’d: 

lstm_1/while/Merge_3Mergelstm_1/while/Enter_3lstm_1/while/NextIteration_3*
T0*
N*)
_output_shapes
:’’’’’’’’’d: 
g
lstm_1/while/LessLesslstm_1/while/Mergelstm_1/while/Less/Enter*
T0*
_output_shapes
: 
³
lstm_1/while/Less/EnterEnterlstm_1/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context
L
lstm_1/while/LoopCondLoopCondlstm_1/while/Less*
_output_shapes
: 

lstm_1/while/SwitchSwitchlstm_1/while/Mergelstm_1/while/LoopCond*%
_class
loc:@lstm_1/while/Merge*
_output_shapes
: : *
T0

lstm_1/while/Switch_1Switchlstm_1/while/Merge_1lstm_1/while/LoopCond*
T0*'
_class
loc:@lstm_1/while/Merge_1*
_output_shapes
: : 
ŗ
lstm_1/while/Switch_2Switchlstm_1/while/Merge_2lstm_1/while/LoopCond*:
_output_shapes(
&:’’’’’’’’’d:’’’’’’’’’d*
T0*'
_class
loc:@lstm_1/while/Merge_2
ŗ
lstm_1/while/Switch_3Switchlstm_1/while/Merge_3lstm_1/while/LoopCond*:
_output_shapes(
&:’’’’’’’’’d:’’’’’’’’’d*
T0*'
_class
loc:@lstm_1/while/Merge_3
Y
lstm_1/while/IdentityIdentitylstm_1/while/Switch:1*
_output_shapes
: *
T0
]
lstm_1/while/Identity_1Identitylstm_1/while/Switch_1:1*
T0*
_output_shapes
: 
n
lstm_1/while/Identity_2Identitylstm_1/while/Switch_2:1*'
_output_shapes
:’’’’’’’’’d*
T0
n
lstm_1/while/Identity_3Identitylstm_1/while/Switch_3:1*'
_output_shapes
:’’’’’’’’’d*
T0
Ī
lstm_1/while/TensorArrayReadV3TensorArrayReadV3$lstm_1/while/TensorArrayReadV3/Enterlstm_1/while/Identity&lstm_1/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:’’’’’’’’’d
Ā
$lstm_1/while/TensorArrayReadV3/EnterEnterlstm_1/TensorArray_1*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
ķ
&lstm_1/while/TensorArrayReadV3/Enter_1EnterAlstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context

lstm_1/while/ReadVariableOpReadVariableOp!lstm_1/while/ReadVariableOp/Enter^lstm_1/while/Identity*
dtype0*
_output_shapes
:	d
“
!lstm_1/while/ReadVariableOp/EnterEnterlstm_1/kernel*
_output_shapes
: **

frame_namelstm_1/while/while_context*
T0*
is_constant(*
parallel_iterations 

 lstm_1/while/strided_slice/stackConst^lstm_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:

"lstm_1/while/strided_slice/stack_1Const^lstm_1/while/Identity*
_output_shapes
:*
valueB"    d   *
dtype0

"lstm_1/while/strided_slice/stack_2Const^lstm_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Ė
lstm_1/while/strided_sliceStridedSlicelstm_1/while/ReadVariableOp lstm_1/while/strided_slice/stack"lstm_1/while/strided_slice/stack_1"lstm_1/while/strided_slice/stack_2*
new_axis_mask *
end_mask*
_output_shapes

:dd*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask
±
lstm_1/while/MatMulMatMullstm_1/while/TensorArrayReadV3lstm_1/while/strided_slice*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 

lstm_1/while/ReadVariableOp_1ReadVariableOp!lstm_1/while/ReadVariableOp/Enter^lstm_1/while/Identity*
dtype0*
_output_shapes
:	d

"lstm_1/while/strided_slice_1/stackConst^lstm_1/while/Identity*
valueB"    d   *
dtype0*
_output_shapes
:

$lstm_1/while/strided_slice_1/stack_1Const^lstm_1/while/Identity*
valueB"    Č   *
dtype0*
_output_shapes
:

$lstm_1/while/strided_slice_1/stack_2Const^lstm_1/while/Identity*
_output_shapes
:*
valueB"      *
dtype0
Õ
lstm_1/while/strided_slice_1StridedSlicelstm_1/while/ReadVariableOp_1"lstm_1/while/strided_slice_1/stack$lstm_1/while/strided_slice_1/stack_1$lstm_1/while/strided_slice_1/stack_2*
_output_shapes

:dd*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
µ
lstm_1/while/MatMul_1MatMullstm_1/while/TensorArrayReadV3lstm_1/while/strided_slice_1*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( *
T0

lstm_1/while/ReadVariableOp_2ReadVariableOp!lstm_1/while/ReadVariableOp/Enter^lstm_1/while/Identity*
dtype0*
_output_shapes
:	d

"lstm_1/while/strided_slice_2/stackConst^lstm_1/while/Identity*
valueB"    Č   *
dtype0*
_output_shapes
:

$lstm_1/while/strided_slice_2/stack_1Const^lstm_1/while/Identity*
valueB"    ,  *
dtype0*
_output_shapes
:

$lstm_1/while/strided_slice_2/stack_2Const^lstm_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Õ
lstm_1/while/strided_slice_2StridedSlicelstm_1/while/ReadVariableOp_2"lstm_1/while/strided_slice_2/stack$lstm_1/while/strided_slice_2/stack_1$lstm_1/while/strided_slice_2/stack_2*
_output_shapes

:dd*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
µ
lstm_1/while/MatMul_2MatMullstm_1/while/TensorArrayReadV3lstm_1/while/strided_slice_2*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( *
T0

lstm_1/while/ReadVariableOp_3ReadVariableOp!lstm_1/while/ReadVariableOp/Enter^lstm_1/while/Identity*
dtype0*
_output_shapes
:	d

"lstm_1/while/strided_slice_3/stackConst^lstm_1/while/Identity*
valueB"    ,  *
dtype0*
_output_shapes
:

$lstm_1/while/strided_slice_3/stack_1Const^lstm_1/while/Identity*
_output_shapes
:*
valueB"        *
dtype0

$lstm_1/while/strided_slice_3/stack_2Const^lstm_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Õ
lstm_1/while/strided_slice_3StridedSlicelstm_1/while/ReadVariableOp_3"lstm_1/while/strided_slice_3/stack$lstm_1/while/strided_slice_3/stack_1$lstm_1/while/strided_slice_3/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:dd*
Index0*
T0
µ
lstm_1/while/MatMul_3MatMullstm_1/while/TensorArrayReadV3lstm_1/while/strided_slice_3*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( *
T0

lstm_1/while/ReadVariableOp_4ReadVariableOp#lstm_1/while/ReadVariableOp_4/Enter^lstm_1/while/Identity*
_output_shapes	
:*
dtype0
“
#lstm_1/while/ReadVariableOp_4/EnterEnterlstm_1/bias*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context*
T0

"lstm_1/while/strided_slice_4/stackConst^lstm_1/while/Identity*
valueB: *
dtype0*
_output_shapes
:

$lstm_1/while/strided_slice_4/stack_1Const^lstm_1/while/Identity*
valueB:d*
dtype0*
_output_shapes
:

$lstm_1/while/strided_slice_4/stack_2Const^lstm_1/while/Identity*
valueB:*
dtype0*
_output_shapes
:
Ń
lstm_1/while/strided_slice_4StridedSlicelstm_1/while/ReadVariableOp_4"lstm_1/while/strided_slice_4/stack$lstm_1/while/strided_slice_4/stack_1$lstm_1/while/strided_slice_4/stack_2*
_output_shapes
:d*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask 

lstm_1/while/BiasAddBiasAddlstm_1/while/MatMullstm_1/while/strided_slice_4*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’d*
T0

lstm_1/while/ReadVariableOp_5ReadVariableOp#lstm_1/while/ReadVariableOp_4/Enter^lstm_1/while/Identity*
dtype0*
_output_shapes	
:

"lstm_1/while/strided_slice_5/stackConst^lstm_1/while/Identity*
_output_shapes
:*
valueB:d*
dtype0

$lstm_1/while/strided_slice_5/stack_1Const^lstm_1/while/Identity*
valueB:Č*
dtype0*
_output_shapes
:

$lstm_1/while/strided_slice_5/stack_2Const^lstm_1/while/Identity*
_output_shapes
:*
valueB:*
dtype0
Ń
lstm_1/while/strided_slice_5StridedSlicelstm_1/while/ReadVariableOp_5"lstm_1/while/strided_slice_5/stack$lstm_1/while/strided_slice_5/stack_1$lstm_1/while/strided_slice_5/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:d*
T0*
Index0

lstm_1/while/BiasAdd_1BiasAddlstm_1/while/MatMul_1lstm_1/while/strided_slice_5*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’d

lstm_1/while/ReadVariableOp_6ReadVariableOp#lstm_1/while/ReadVariableOp_4/Enter^lstm_1/while/Identity*
_output_shapes	
:*
dtype0

"lstm_1/while/strided_slice_6/stackConst^lstm_1/while/Identity*
valueB:Č*
dtype0*
_output_shapes
:

$lstm_1/while/strided_slice_6/stack_1Const^lstm_1/while/Identity*
valueB:¬*
dtype0*
_output_shapes
:

$lstm_1/while/strided_slice_6/stack_2Const^lstm_1/while/Identity*
valueB:*
dtype0*
_output_shapes
:
Ń
lstm_1/while/strided_slice_6StridedSlicelstm_1/while/ReadVariableOp_6"lstm_1/while/strided_slice_6/stack$lstm_1/while/strided_slice_6/stack_1$lstm_1/while/strided_slice_6/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:d*
Index0*
T0*
shrink_axis_mask 

lstm_1/while/BiasAdd_2BiasAddlstm_1/while/MatMul_2lstm_1/while/strided_slice_6*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’d

lstm_1/while/ReadVariableOp_7ReadVariableOp#lstm_1/while/ReadVariableOp_4/Enter^lstm_1/while/Identity*
dtype0*
_output_shapes	
:

"lstm_1/while/strided_slice_7/stackConst^lstm_1/while/Identity*
valueB:¬*
dtype0*
_output_shapes
:

$lstm_1/while/strided_slice_7/stack_1Const^lstm_1/while/Identity*
valueB: *
dtype0*
_output_shapes
:

$lstm_1/while/strided_slice_7/stack_2Const^lstm_1/while/Identity*
valueB:*
dtype0*
_output_shapes
:
Ń
lstm_1/while/strided_slice_7StridedSlicelstm_1/while/ReadVariableOp_7"lstm_1/while/strided_slice_7/stack$lstm_1/while/strided_slice_7/stack_1$lstm_1/while/strided_slice_7/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:d*
Index0*
T0*
shrink_axis_mask 

lstm_1/while/BiasAdd_3BiasAddlstm_1/while/MatMul_3lstm_1/while/strided_slice_7*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’d

lstm_1/while/ReadVariableOp_8ReadVariableOp#lstm_1/while/ReadVariableOp_8/Enter^lstm_1/while/Identity*
dtype0*
_output_shapes
:	d
Ą
#lstm_1/while/ReadVariableOp_8/EnterEnterlstm_1/recurrent_kernel*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context

"lstm_1/while/strided_slice_8/stackConst^lstm_1/while/Identity*
_output_shapes
:*
valueB"        *
dtype0

$lstm_1/while/strided_slice_8/stack_1Const^lstm_1/while/Identity*
valueB"    d   *
dtype0*
_output_shapes
:

$lstm_1/while/strided_slice_8/stack_2Const^lstm_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Õ
lstm_1/while/strided_slice_8StridedSlicelstm_1/while/ReadVariableOp_8"lstm_1/while/strided_slice_8/stack$lstm_1/while/strided_slice_8/stack_1$lstm_1/while/strided_slice_8/stack_2*
_output_shapes

:dd*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
®
lstm_1/while/MatMul_4MatMullstm_1/while/Identity_2lstm_1/while/strided_slice_8*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( *
T0
v
lstm_1/while/addAddlstm_1/while/BiasAddlstm_1/while/MatMul_4*'
_output_shapes
:’’’’’’’’’d*
T0
o
lstm_1/while/mul/xConst^lstm_1/while/Identity*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 
o
lstm_1/while/mulMullstm_1/while/mul/xlstm_1/while/add*'
_output_shapes
:’’’’’’’’’d*
T0
q
lstm_1/while/add_1/yConst^lstm_1/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
s
lstm_1/while/add_1Addlstm_1/while/mullstm_1/while/add_1/y*'
_output_shapes
:’’’’’’’’’d*
T0
o
lstm_1/while/ConstConst^lstm_1/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
q
lstm_1/while/Const_1Const^lstm_1/while/Identity*
_output_shapes
: *
valueB
 *  ?*
dtype0

"lstm_1/while/clip_by_value/MinimumMinimumlstm_1/while/add_1lstm_1/while/Const_1*
T0*'
_output_shapes
:’’’’’’’’’d

lstm_1/while/clip_by_valueMaximum"lstm_1/while/clip_by_value/Minimumlstm_1/while/Const*'
_output_shapes
:’’’’’’’’’d*
T0

lstm_1/while/ReadVariableOp_9ReadVariableOp#lstm_1/while/ReadVariableOp_8/Enter^lstm_1/while/Identity*
_output_shapes
:	d*
dtype0

"lstm_1/while/strided_slice_9/stackConst^lstm_1/while/Identity*
valueB"    d   *
dtype0*
_output_shapes
:

$lstm_1/while/strided_slice_9/stack_1Const^lstm_1/while/Identity*
_output_shapes
:*
valueB"    Č   *
dtype0

$lstm_1/while/strided_slice_9/stack_2Const^lstm_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Õ
lstm_1/while/strided_slice_9StridedSlicelstm_1/while/ReadVariableOp_9"lstm_1/while/strided_slice_9/stack$lstm_1/while/strided_slice_9/stack_1$lstm_1/while/strided_slice_9/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:dd
®
lstm_1/while/MatMul_5MatMullstm_1/while/Identity_2lstm_1/while/strided_slice_9*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
z
lstm_1/while/add_2Addlstm_1/while/BiasAdd_1lstm_1/while/MatMul_5*'
_output_shapes
:’’’’’’’’’d*
T0
q
lstm_1/while/mul_1/xConst^lstm_1/while/Identity*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 
u
lstm_1/while/mul_1Mullstm_1/while/mul_1/xlstm_1/while/add_2*
T0*'
_output_shapes
:’’’’’’’’’d
q
lstm_1/while/add_3/yConst^lstm_1/while/Identity*
_output_shapes
: *
valueB
 *   ?*
dtype0
u
lstm_1/while/add_3Addlstm_1/while/mul_1lstm_1/while/add_3/y*
T0*'
_output_shapes
:’’’’’’’’’d
q
lstm_1/while/Const_2Const^lstm_1/while/Identity*
_output_shapes
: *
valueB
 *    *
dtype0
q
lstm_1/while/Const_3Const^lstm_1/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$lstm_1/while/clip_by_value_1/MinimumMinimumlstm_1/while/add_3lstm_1/while/Const_3*
T0*'
_output_shapes
:’’’’’’’’’d

lstm_1/while/clip_by_value_1Maximum$lstm_1/while/clip_by_value_1/Minimumlstm_1/while/Const_2*'
_output_shapes
:’’’’’’’’’d*
T0

lstm_1/while/mul_2Mullstm_1/while/clip_by_value_1lstm_1/while/Identity_3*
T0*'
_output_shapes
:’’’’’’’’’d

lstm_1/while/ReadVariableOp_10ReadVariableOp#lstm_1/while/ReadVariableOp_8/Enter^lstm_1/while/Identity*
_output_shapes
:	d*
dtype0

#lstm_1/while/strided_slice_10/stackConst^lstm_1/while/Identity*
valueB"    Č   *
dtype0*
_output_shapes
:

%lstm_1/while/strided_slice_10/stack_1Const^lstm_1/while/Identity*
valueB"    ,  *
dtype0*
_output_shapes
:

%lstm_1/while/strided_slice_10/stack_2Const^lstm_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Ś
lstm_1/while/strided_slice_10StridedSlicelstm_1/while/ReadVariableOp_10#lstm_1/while/strided_slice_10/stack%lstm_1/while/strided_slice_10/stack_1%lstm_1/while/strided_slice_10/stack_2*
new_axis_mask *
end_mask*
_output_shapes

:dd*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask
Æ
lstm_1/while/MatMul_6MatMullstm_1/while/Identity_2lstm_1/while/strided_slice_10*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
z
lstm_1/while/add_4Addlstm_1/while/BiasAdd_2lstm_1/while/MatMul_6*
T0*'
_output_shapes
:’’’’’’’’’d
_
lstm_1/while/TanhTanhlstm_1/while/add_4*'
_output_shapes
:’’’’’’’’’d*
T0
z
lstm_1/while/mul_3Mullstm_1/while/clip_by_valuelstm_1/while/Tanh*'
_output_shapes
:’’’’’’’’’d*
T0
s
lstm_1/while/add_5Addlstm_1/while/mul_2lstm_1/while/mul_3*
T0*'
_output_shapes
:’’’’’’’’’d

lstm_1/while/ReadVariableOp_11ReadVariableOp#lstm_1/while/ReadVariableOp_8/Enter^lstm_1/while/Identity*
_output_shapes
:	d*
dtype0

#lstm_1/while/strided_slice_11/stackConst^lstm_1/while/Identity*
_output_shapes
:*
valueB"    ,  *
dtype0

%lstm_1/while/strided_slice_11/stack_1Const^lstm_1/while/Identity*
valueB"        *
dtype0*
_output_shapes
:

%lstm_1/while/strided_slice_11/stack_2Const^lstm_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
Ś
lstm_1/while/strided_slice_11StridedSlicelstm_1/while/ReadVariableOp_11#lstm_1/while/strided_slice_11/stack%lstm_1/while/strided_slice_11/stack_1%lstm_1/while/strided_slice_11/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:dd*
T0*
Index0
Æ
lstm_1/while/MatMul_7MatMullstm_1/while/Identity_2lstm_1/while/strided_slice_11*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( 
z
lstm_1/while/add_6Addlstm_1/while/BiasAdd_3lstm_1/while/MatMul_7*
T0*'
_output_shapes
:’’’’’’’’’d
q
lstm_1/while/mul_4/xConst^lstm_1/while/Identity*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 
u
lstm_1/while/mul_4Mullstm_1/while/mul_4/xlstm_1/while/add_6*'
_output_shapes
:’’’’’’’’’d*
T0
q
lstm_1/while/add_7/yConst^lstm_1/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
u
lstm_1/while/add_7Addlstm_1/while/mul_4lstm_1/while/add_7/y*
T0*'
_output_shapes
:’’’’’’’’’d
q
lstm_1/while/Const_4Const^lstm_1/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
q
lstm_1/while/Const_5Const^lstm_1/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$lstm_1/while/clip_by_value_2/MinimumMinimumlstm_1/while/add_7lstm_1/while/Const_5*
T0*'
_output_shapes
:’’’’’’’’’d

lstm_1/while/clip_by_value_2Maximum$lstm_1/while/clip_by_value_2/Minimumlstm_1/while/Const_4*'
_output_shapes
:’’’’’’’’’d*
T0
a
lstm_1/while/Tanh_1Tanhlstm_1/while/add_5*'
_output_shapes
:’’’’’’’’’d*
T0
~
lstm_1/while/mul_5Mullstm_1/while/clip_by_value_2lstm_1/while/Tanh_1*'
_output_shapes
:’’’’’’’’’d*
T0

0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/Enterlstm_1/while/Identitylstm_1/while/mul_5lstm_1/while/Identity_1*
T0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
: 
ł
6lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlstm_1/TensorArray*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
n
lstm_1/while/add_8/yConst^lstm_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
g
lstm_1/while/add_8Addlstm_1/while/Identitylstm_1/while/add_8/y*
_output_shapes
: *
T0
`
lstm_1/while/NextIterationNextIterationlstm_1/while/add_8*
_output_shapes
: *
T0

lstm_1/while/NextIteration_1NextIteration0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
s
lstm_1/while/NextIteration_2NextIterationlstm_1/while/mul_5*
T0*'
_output_shapes
:’’’’’’’’’d
s
lstm_1/while/NextIteration_3NextIterationlstm_1/while/add_5*'
_output_shapes
:’’’’’’’’’d*
T0
O
lstm_1/while/ExitExitlstm_1/while/Switch*
_output_shapes
: *
T0
S
lstm_1/while/Exit_1Exitlstm_1/while/Switch_1*
T0*
_output_shapes
: 
d
lstm_1/while/Exit_2Exitlstm_1/while/Switch_2*'
_output_shapes
:’’’’’’’’’d*
T0
d
lstm_1/while/Exit_3Exitlstm_1/while/Switch_3*'
_output_shapes
:’’’’’’’’’d*
T0
¦
)lstm_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lstm_1/TensorArraylstm_1/while/Exit_1*%
_class
loc:@lstm_1/TensorArray*
_output_shapes
: 

#lstm_1/TensorArrayStack/range/startConst*%
_class
loc:@lstm_1/TensorArray*
value	B : *
dtype0*
_output_shapes
: 

#lstm_1/TensorArrayStack/range/deltaConst*%
_class
loc:@lstm_1/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
ó
lstm_1/TensorArrayStack/rangeRange#lstm_1/TensorArrayStack/range/start)lstm_1/TensorArrayStack/TensorArraySizeV3#lstm_1/TensorArrayStack/range/delta*%
_class
loc:@lstm_1/TensorArray*#
_output_shapes
:’’’’’’’’’*

Tidx0

+lstm_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lstm_1/TensorArraylstm_1/TensorArrayStack/rangelstm_1/while/Exit_1*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d*$
element_shape:’’’’’’’’’d*%
_class
loc:@lstm_1/TensorArray*
dtype0
N
lstm_1/sub/yConst*
_output_shapes
: *
value	B :*
dtype0
S

lstm_1/subSublstm_1/while/Exitlstm_1/sub/y*
_output_shapes
: *
T0

lstm_1/TensorArrayReadV3TensorArrayReadV3lstm_1/TensorArray
lstm_1/sublstm_1/while/Exit_1*
dtype0*'
_output_shapes
:’’’’’’’’’d
l
lstm_1/transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0
±
lstm_1/transpose_1	Transpose+lstm_1/TensorArrayStack/TensorArrayGatherV3lstm_1/transpose_1/perm*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’d*
Tperm0
Y
concatenate/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
±
concatenate/concatConcatV2lstm/TensorArrayReadV3lstm_1/TensorArrayReadV3concatenate/concat/axis*(
_output_shapes
:’’’’’’’’’Č*

Tidx0*
T0*
N
~
dropout_1/cond/SwitchSwitchdropout/keras_learning_phasedropout/keras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
_output_shapes
: *
T0

a
dropout_1/cond/pred_idIdentitydropout/keras_learning_phase*
T0
*
_output_shapes
: 

 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 

dropout_1/cond/dropout/ShapeShape%dropout_1/cond/dropout/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0
Ē
#dropout_1/cond/dropout/Shape/SwitchSwitchconcatenate/concatdropout_1/cond/pred_id*
T0*%
_class
loc:@concatenate/concat*<
_output_shapes*
(:’’’’’’’’’Č:’’’’’’’’’Č

)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
»
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
T0*
dtype0*(
_output_shapes
:’’’’’’’’’Č*
seed2 *

seed 
§
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ć
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*(
_output_shapes
:’’’’’’’’’Č*
T0
µ
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*(
_output_shapes
:’’’’’’’’’Č*
T0

dropout_1/cond/dropout/addAdd dropout_1/cond/dropout/keep_prob%dropout_1/cond/dropout/random_uniform*
T0*(
_output_shapes
:’’’’’’’’’Č
t
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*
T0*(
_output_shapes
:’’’’’’’’’Č
”
dropout_1/cond/dropout/divRealDiv%dropout_1/cond/dropout/Shape/Switch:1 dropout_1/cond/dropout/keep_prob*(
_output_shapes
:’’’’’’’’’Č*
T0

dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*
T0*(
_output_shapes
:’’’’’’’’’Č
v
dropout_1/cond/IdentityIdentitydropout_1/cond/Identity/Switch*
T0*(
_output_shapes
:’’’’’’’’’Č
Ā
dropout_1/cond/Identity/SwitchSwitchconcatenate/concatdropout_1/cond/pred_id*
T0*%
_class
loc:@concatenate/concat*<
_output_shapes*
(:’’’’’’’’’Č:’’’’’’’’’Č

dropout_1/cond/MergeMergedropout_1/cond/Identitydropout_1/cond/dropout/mul*
T0*
N**
_output_shapes
:’’’’’’’’’Č: 
X
lambda_1/ShapeShape
word_input*
_output_shapes
:*
T0*
out_type0
f
lambda_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
h
lambda_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
lambda_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¦
lambda_1/strided_sliceStridedSlicelambda_1/Shapelambda_1/strided_slice/stacklambda_1/strided_slice/stack_1lambda_1/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
c
lambda_1/Reshape/shape/0Const*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 
[
lambda_1/Reshape/shape/2Const*
value
B :Č*
dtype0*
_output_shapes
: 
¤
lambda_1/Reshape/shapePacklambda_1/Reshape/shape/0lambda_1/strided_slicelambda_1/Reshape/shape/2*
T0*

axis *
N*
_output_shapes
:

lambda_1/ReshapeReshapedropout_1/cond/Mergelambda_1/Reshape/shape*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’Č*
T0*
Tshape0
[
concatenate_1/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
É
concatenate_1/concatConcatV2%embedding/embedding_lookup/Identity_2lambda_1/Reshapeconcatenate_1/concat/axis*

Tidx0*
T0*
N*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’ō
~
dropout_2/cond/SwitchSwitchdropout/keras_learning_phasedropout/keras_learning_phase*
_output_shapes
: : *
T0

]
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
T0
*
_output_shapes
: 
a
dropout_2/cond/pred_idIdentitydropout/keras_learning_phase*
T0
*
_output_shapes
: 

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
_output_shapes
: *
valueB
 *   ?*
dtype0

dropout_2/cond/dropout/ShapeShape%dropout_2/cond/dropout/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0
å
#dropout_2/cond/dropout/Shape/SwitchSwitchconcatenate_1/concatdropout_2/cond/pred_id*'
_class
loc:@concatenate_1/concat*V
_output_shapesD
B:’’’’’’’’’’’’’’’’’’ō:’’’’’’’’’’’’’’’’’’ō*
T0

)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Č
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
T0*
dtype0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’ō*
seed2 *

seed 
§
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Š
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’ō
Ā
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’ō
Ŗ
dropout_2/cond/dropout/addAdd dropout_2/cond/dropout/keep_prob%dropout_2/cond/dropout/random_uniform*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’ō*
T0

dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’ō*
T0
®
dropout_2/cond/dropout/divRealDiv%dropout_2/cond/dropout/Shape/Switch:1 dropout_2/cond/dropout/keep_prob*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’ō*
T0

dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’ō*
T0

dropout_2/cond/IdentityIdentitydropout_2/cond/Identity/Switch*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’ō
ą
dropout_2/cond/Identity/SwitchSwitchconcatenate_1/concatdropout_2/cond/pred_id*V
_output_shapesD
B:’’’’’’’’’’’’’’’’’’ō:’’’’’’’’’’’’’’’’’’ō*
T0*'
_class
loc:@concatenate_1/concat

dropout_2/cond/MergeMergedropout_2/cond/Identitydropout_2/cond/dropout/mul*
T0*
N*7
_output_shapes%
#:’’’’’’’’’’’’’’’’’’ō: 
Ķ
Dbidirectional/forward_lstm_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*6
_class,
*(loc:@bidirectional/forward_lstm_2/kernel*
valueB"ō  °  *
dtype0
æ
Bbidirectional/forward_lstm_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *6
_class,
*(loc:@bidirectional/forward_lstm_2/kernel*
valueB
 *²Vs½*
dtype0
æ
Bbidirectional/forward_lstm_2/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@bidirectional/forward_lstm_2/kernel*
valueB
 *²Vs=*
dtype0*
_output_shapes
: 
¬
Lbidirectional/forward_lstm_2/kernel/Initializer/random_uniform/RandomUniformRandomUniformDbidirectional/forward_lstm_2/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
ō°	*

seed *
T0*6
_class,
*(loc:@bidirectional/forward_lstm_2/kernel
Ŗ
Bbidirectional/forward_lstm_2/kernel/Initializer/random_uniform/subSubBbidirectional/forward_lstm_2/kernel/Initializer/random_uniform/maxBbidirectional/forward_lstm_2/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@bidirectional/forward_lstm_2/kernel*
_output_shapes
: 
¾
Bbidirectional/forward_lstm_2/kernel/Initializer/random_uniform/mulMulLbidirectional/forward_lstm_2/kernel/Initializer/random_uniform/RandomUniformBbidirectional/forward_lstm_2/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
ō°	*
T0*6
_class,
*(loc:@bidirectional/forward_lstm_2/kernel
°
>bidirectional/forward_lstm_2/kernel/Initializer/random_uniformAddBbidirectional/forward_lstm_2/kernel/Initializer/random_uniform/mulBbidirectional/forward_lstm_2/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@bidirectional/forward_lstm_2/kernel* 
_output_shapes
:
ō°	*
T0
ķ
#bidirectional/forward_lstm_2/kernelVarHandleOp*
	container *
shape:
ō°	*
dtype0*
_output_shapes
: *4
shared_name%#bidirectional/forward_lstm_2/kernel*6
_class,
*(loc:@bidirectional/forward_lstm_2/kernel

Dbidirectional/forward_lstm_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp#bidirectional/forward_lstm_2/kernel*
_output_shapes
: 
č
*bidirectional/forward_lstm_2/kernel/AssignAssignVariableOp#bidirectional/forward_lstm_2/kernel>bidirectional/forward_lstm_2/kernel/Initializer/random_uniform*6
_class,
*(loc:@bidirectional/forward_lstm_2/kernel*
dtype0
Õ
7bidirectional/forward_lstm_2/kernel/Read/ReadVariableOpReadVariableOp#bidirectional/forward_lstm_2/kernel*6
_class,
*(loc:@bidirectional/forward_lstm_2/kernel*
dtype0* 
_output_shapes
:
ō°	
ą
Mbidirectional/forward_lstm_2/recurrent_kernel/Initializer/random_normal/shapeConst*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel*
valueB"°  ,  *
dtype0*
_output_shapes
:
Ó
Lbidirectional/forward_lstm_2/recurrent_kernel/Initializer/random_normal/meanConst*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Õ
Nbidirectional/forward_lstm_2/recurrent_kernel/Initializer/random_normal/stddevConst*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ö
\bidirectional/forward_lstm_2/recurrent_kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalMbidirectional/forward_lstm_2/recurrent_kernel/Initializer/random_normal/shape*
dtype0* 
_output_shapes
:
°	¬*

seed *
T0*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel*
seed2 
ķ
Kbidirectional/forward_lstm_2/recurrent_kernel/Initializer/random_normal/mulMul\bidirectional/forward_lstm_2/recurrent_kernel/Initializer/random_normal/RandomStandardNormalNbidirectional/forward_lstm_2/recurrent_kernel/Initializer/random_normal/stddev* 
_output_shapes
:
°	¬*
T0*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel
Ö
Gbidirectional/forward_lstm_2/recurrent_kernel/Initializer/random_normalAddKbidirectional/forward_lstm_2/recurrent_kernel/Initializer/random_normal/mulLbidirectional/forward_lstm_2/recurrent_kernel/Initializer/random_normal/mean*
T0*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel* 
_output_shapes
:
°	¬

<bidirectional/forward_lstm_2/recurrent_kernel/Initializer/QrQrGbidirectional/forward_lstm_2/recurrent_kernel/Initializer/random_normal*
full_matrices( *
T0*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel*,
_output_shapes
:
°	¬:
¬¬
ö
Bbidirectional/forward_lstm_2/recurrent_kernel/Initializer/DiagPartDiagPart>bidirectional/forward_lstm_2/recurrent_kernel/Initializer/Qr:1*
_output_shapes	
:¬*
T0*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel
ņ
>bidirectional/forward_lstm_2/recurrent_kernel/Initializer/SignSignBbidirectional/forward_lstm_2/recurrent_kernel/Initializer/DiagPart*
_output_shapes	
:¬*
T0*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel
Æ
=bidirectional/forward_lstm_2/recurrent_kernel/Initializer/mulMul<bidirectional/forward_lstm_2/recurrent_kernel/Initializer/Qr>bidirectional/forward_lstm_2/recurrent_kernel/Initializer/Sign* 
_output_shapes
:
°	¬*
T0*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel
ģ
Ybidirectional/forward_lstm_2/recurrent_kernel/Initializer/matrix_transpose/transpose/permConst*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel*
valueB"       *
dtype0*
_output_shapes
:
õ
Tbidirectional/forward_lstm_2/recurrent_kernel/Initializer/matrix_transpose/transpose	Transpose=bidirectional/forward_lstm_2/recurrent_kernel/Initializer/mulYbidirectional/forward_lstm_2/recurrent_kernel/Initializer/matrix_transpose/transpose/perm*
T0*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel* 
_output_shapes
:
¬°	*
Tperm0
Ś
Gbidirectional/forward_lstm_2/recurrent_kernel/Initializer/Reshape/shapeConst*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel*
valueB",  °  *
dtype0*
_output_shapes
:
ę
Abidirectional/forward_lstm_2/recurrent_kernel/Initializer/ReshapeReshapeTbidirectional/forward_lstm_2/recurrent_kernel/Initializer/matrix_transpose/transposeGbidirectional/forward_lstm_2/recurrent_kernel/Initializer/Reshape/shape*
T0*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel*
Tshape0* 
_output_shapes
:
¬°	
Č
Abidirectional/forward_lstm_2/recurrent_kernel/Initializer/mul_1/xConst*
_output_shapes
: *@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel*
valueB
 *  ?*
dtype0
¹
?bidirectional/forward_lstm_2/recurrent_kernel/Initializer/mul_1MulAbidirectional/forward_lstm_2/recurrent_kernel/Initializer/mul_1/xAbidirectional/forward_lstm_2/recurrent_kernel/Initializer/Reshape* 
_output_shapes
:
¬°	*
T0*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel

-bidirectional/forward_lstm_2/recurrent_kernelVarHandleOp*
	container *
shape:
¬°	*
dtype0*
_output_shapes
: *>
shared_name/-bidirectional/forward_lstm_2/recurrent_kernel*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel
«
Nbidirectional/forward_lstm_2/recurrent_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp-bidirectional/forward_lstm_2/recurrent_kernel*
_output_shapes
: 

4bidirectional/forward_lstm_2/recurrent_kernel/AssignAssignVariableOp-bidirectional/forward_lstm_2/recurrent_kernel?bidirectional/forward_lstm_2/recurrent_kernel/Initializer/mul_1*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel*
dtype0
ó
Abidirectional/forward_lstm_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp-bidirectional/forward_lstm_2/recurrent_kernel*@
_class6
42loc:@bidirectional/forward_lstm_2/recurrent_kernel*
dtype0* 
_output_shapes
:
¬°	
ø
3bidirectional/forward_lstm_2/bias/Initializer/zerosConst*4
_class*
(&loc:@bidirectional/forward_lstm_2/bias*
valueB¬*    *
dtype0*
_output_shapes	
:¬
·
2bidirectional/forward_lstm_2/bias/Initializer/onesConst*4
_class*
(&loc:@bidirectional/forward_lstm_2/bias*
valueB¬*  ?*
dtype0*
_output_shapes	
:¬
ŗ
5bidirectional/forward_lstm_2/bias/Initializer/zeros_1Const*4
_class*
(&loc:@bidirectional/forward_lstm_2/bias*
valueBŲ*    *
dtype0*
_output_shapes	
:Ų
±
9bidirectional/forward_lstm_2/bias/Initializer/concat/axisConst*
_output_shapes
: *4
_class*
(&loc:@bidirectional/forward_lstm_2/bias*
value	B : *
dtype0

4bidirectional/forward_lstm_2/bias/Initializer/concatConcatV23bidirectional/forward_lstm_2/bias/Initializer/zeros2bidirectional/forward_lstm_2/bias/Initializer/ones5bidirectional/forward_lstm_2/bias/Initializer/zeros_19bidirectional/forward_lstm_2/bias/Initializer/concat/axis*
_output_shapes	
:°	*

Tidx0*
T0*4
_class*
(&loc:@bidirectional/forward_lstm_2/bias*
N
ā
!bidirectional/forward_lstm_2/biasVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!bidirectional/forward_lstm_2/bias*4
_class*
(&loc:@bidirectional/forward_lstm_2/bias*
	container *
shape:°	

Bbidirectional/forward_lstm_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp!bidirectional/forward_lstm_2/bias*
_output_shapes
: 
Ų
(bidirectional/forward_lstm_2/bias/AssignAssignVariableOp!bidirectional/forward_lstm_2/bias4bidirectional/forward_lstm_2/bias/Initializer/concat*4
_class*
(&loc:@bidirectional/forward_lstm_2/bias*
dtype0
Ź
5bidirectional/forward_lstm_2/bias/Read/ReadVariableOpReadVariableOp!bidirectional/forward_lstm_2/bias*4
_class*
(&loc:@bidirectional/forward_lstm_2/bias*
dtype0*
_output_shapes	
:°	
Ļ
Ebidirectional/backward_lstm_2/kernel/Initializer/random_uniform/shapeConst*7
_class-
+)loc:@bidirectional/backward_lstm_2/kernel*
valueB"ō  °  *
dtype0*
_output_shapes
:
Į
Cbidirectional/backward_lstm_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *7
_class-
+)loc:@bidirectional/backward_lstm_2/kernel*
valueB
 *²Vs½*
dtype0
Į
Cbidirectional/backward_lstm_2/kernel/Initializer/random_uniform/maxConst*7
_class-
+)loc:@bidirectional/backward_lstm_2/kernel*
valueB
 *²Vs=*
dtype0*
_output_shapes
: 
Æ
Mbidirectional/backward_lstm_2/kernel/Initializer/random_uniform/RandomUniformRandomUniformEbidirectional/backward_lstm_2/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
ō°	*

seed *
T0*7
_class-
+)loc:@bidirectional/backward_lstm_2/kernel*
seed2 
®
Cbidirectional/backward_lstm_2/kernel/Initializer/random_uniform/subSubCbidirectional/backward_lstm_2/kernel/Initializer/random_uniform/maxCbidirectional/backward_lstm_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*7
_class-
+)loc:@bidirectional/backward_lstm_2/kernel
Ā
Cbidirectional/backward_lstm_2/kernel/Initializer/random_uniform/mulMulMbidirectional/backward_lstm_2/kernel/Initializer/random_uniform/RandomUniformCbidirectional/backward_lstm_2/kernel/Initializer/random_uniform/sub*
T0*7
_class-
+)loc:@bidirectional/backward_lstm_2/kernel* 
_output_shapes
:
ō°	
“
?bidirectional/backward_lstm_2/kernel/Initializer/random_uniformAddCbidirectional/backward_lstm_2/kernel/Initializer/random_uniform/mulCbidirectional/backward_lstm_2/kernel/Initializer/random_uniform/min*
T0*7
_class-
+)loc:@bidirectional/backward_lstm_2/kernel* 
_output_shapes
:
ō°	
š
$bidirectional/backward_lstm_2/kernelVarHandleOp*5
shared_name&$bidirectional/backward_lstm_2/kernel*7
_class-
+)loc:@bidirectional/backward_lstm_2/kernel*
	container *
shape:
ō°	*
dtype0*
_output_shapes
: 

Ebidirectional/backward_lstm_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp$bidirectional/backward_lstm_2/kernel*
_output_shapes
: 
ģ
+bidirectional/backward_lstm_2/kernel/AssignAssignVariableOp$bidirectional/backward_lstm_2/kernel?bidirectional/backward_lstm_2/kernel/Initializer/random_uniform*7
_class-
+)loc:@bidirectional/backward_lstm_2/kernel*
dtype0
Ų
8bidirectional/backward_lstm_2/kernel/Read/ReadVariableOpReadVariableOp$bidirectional/backward_lstm_2/kernel*7
_class-
+)loc:@bidirectional/backward_lstm_2/kernel*
dtype0* 
_output_shapes
:
ō°	
ā
Nbidirectional/backward_lstm_2/recurrent_kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel*
valueB"°  ,  *
dtype0
Õ
Mbidirectional/backward_lstm_2/recurrent_kernel/Initializer/random_normal/meanConst*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
×
Obidirectional/backward_lstm_2/recurrent_kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel*
valueB
 *  ?*
dtype0
Ł
]bidirectional/backward_lstm_2/recurrent_kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalNbidirectional/backward_lstm_2/recurrent_kernel/Initializer/random_normal/shape*
dtype0* 
_output_shapes
:
°	¬*

seed *
T0*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel*
seed2 
ń
Lbidirectional/backward_lstm_2/recurrent_kernel/Initializer/random_normal/mulMul]bidirectional/backward_lstm_2/recurrent_kernel/Initializer/random_normal/RandomStandardNormalObidirectional/backward_lstm_2/recurrent_kernel/Initializer/random_normal/stddev*
T0*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel* 
_output_shapes
:
°	¬
Ś
Hbidirectional/backward_lstm_2/recurrent_kernel/Initializer/random_normalAddLbidirectional/backward_lstm_2/recurrent_kernel/Initializer/random_normal/mulMbidirectional/backward_lstm_2/recurrent_kernel/Initializer/random_normal/mean* 
_output_shapes
:
°	¬*
T0*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel

=bidirectional/backward_lstm_2/recurrent_kernel/Initializer/QrQrHbidirectional/backward_lstm_2/recurrent_kernel/Initializer/random_normal*,
_output_shapes
:
°	¬:
¬¬*
full_matrices( *
T0*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel
ł
Cbidirectional/backward_lstm_2/recurrent_kernel/Initializer/DiagPartDiagPart?bidirectional/backward_lstm_2/recurrent_kernel/Initializer/Qr:1*
T0*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel*
_output_shapes	
:¬
õ
?bidirectional/backward_lstm_2/recurrent_kernel/Initializer/SignSignCbidirectional/backward_lstm_2/recurrent_kernel/Initializer/DiagPart*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel*
_output_shapes	
:¬*
T0
³
>bidirectional/backward_lstm_2/recurrent_kernel/Initializer/mulMul=bidirectional/backward_lstm_2/recurrent_kernel/Initializer/Qr?bidirectional/backward_lstm_2/recurrent_kernel/Initializer/Sign*
T0*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel* 
_output_shapes
:
°	¬
ī
Zbidirectional/backward_lstm_2/recurrent_kernel/Initializer/matrix_transpose/transpose/permConst*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel*
valueB"       *
dtype0*
_output_shapes
:
ł
Ubidirectional/backward_lstm_2/recurrent_kernel/Initializer/matrix_transpose/transpose	Transpose>bidirectional/backward_lstm_2/recurrent_kernel/Initializer/mulZbidirectional/backward_lstm_2/recurrent_kernel/Initializer/matrix_transpose/transpose/perm*
T0*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel* 
_output_shapes
:
¬°	*
Tperm0
Ü
Hbidirectional/backward_lstm_2/recurrent_kernel/Initializer/Reshape/shapeConst*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel*
valueB",  °  *
dtype0*
_output_shapes
:
ź
Bbidirectional/backward_lstm_2/recurrent_kernel/Initializer/ReshapeReshapeUbidirectional/backward_lstm_2/recurrent_kernel/Initializer/matrix_transpose/transposeHbidirectional/backward_lstm_2/recurrent_kernel/Initializer/Reshape/shape*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel*
Tshape0* 
_output_shapes
:
¬°	*
T0
Ź
Bbidirectional/backward_lstm_2/recurrent_kernel/Initializer/mul_1/xConst*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel*
valueB
 *  ?*
dtype0*
_output_shapes
: 
½
@bidirectional/backward_lstm_2/recurrent_kernel/Initializer/mul_1MulBbidirectional/backward_lstm_2/recurrent_kernel/Initializer/mul_1/xBbidirectional/backward_lstm_2/recurrent_kernel/Initializer/Reshape*
T0*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel* 
_output_shapes
:
¬°	

.bidirectional/backward_lstm_2/recurrent_kernelVarHandleOp*
_output_shapes
: *?
shared_name0.bidirectional/backward_lstm_2/recurrent_kernel*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel*
	container *
shape:
¬°	*
dtype0
­
Obidirectional/backward_lstm_2/recurrent_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp.bidirectional/backward_lstm_2/recurrent_kernel*
_output_shapes
: 

5bidirectional/backward_lstm_2/recurrent_kernel/AssignAssignVariableOp.bidirectional/backward_lstm_2/recurrent_kernel@bidirectional/backward_lstm_2/recurrent_kernel/Initializer/mul_1*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel*
dtype0
ö
Bbidirectional/backward_lstm_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp.bidirectional/backward_lstm_2/recurrent_kernel*A
_class7
53loc:@bidirectional/backward_lstm_2/recurrent_kernel*
dtype0* 
_output_shapes
:
¬°	
ŗ
4bidirectional/backward_lstm_2/bias/Initializer/zerosConst*
_output_shapes	
:¬*5
_class+
)'loc:@bidirectional/backward_lstm_2/bias*
valueB¬*    *
dtype0
¹
3bidirectional/backward_lstm_2/bias/Initializer/onesConst*5
_class+
)'loc:@bidirectional/backward_lstm_2/bias*
valueB¬*  ?*
dtype0*
_output_shapes	
:¬
¼
6bidirectional/backward_lstm_2/bias/Initializer/zeros_1Const*
_output_shapes	
:Ų*5
_class+
)'loc:@bidirectional/backward_lstm_2/bias*
valueBŲ*    *
dtype0
³
:bidirectional/backward_lstm_2/bias/Initializer/concat/axisConst*
_output_shapes
: *5
_class+
)'loc:@bidirectional/backward_lstm_2/bias*
value	B : *
dtype0

5bidirectional/backward_lstm_2/bias/Initializer/concatConcatV24bidirectional/backward_lstm_2/bias/Initializer/zeros3bidirectional/backward_lstm_2/bias/Initializer/ones6bidirectional/backward_lstm_2/bias/Initializer/zeros_1:bidirectional/backward_lstm_2/bias/Initializer/concat/axis*

Tidx0*
T0*5
_class+
)'loc:@bidirectional/backward_lstm_2/bias*
N*
_output_shapes	
:°	
å
"bidirectional/backward_lstm_2/biasVarHandleOp*
dtype0*
_output_shapes
: *3
shared_name$"bidirectional/backward_lstm_2/bias*5
_class+
)'loc:@bidirectional/backward_lstm_2/bias*
	container *
shape:°	

Cbidirectional/backward_lstm_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp"bidirectional/backward_lstm_2/bias*
_output_shapes
: 
Ü
)bidirectional/backward_lstm_2/bias/AssignAssignVariableOp"bidirectional/backward_lstm_2/bias5bidirectional/backward_lstm_2/bias/Initializer/concat*5
_class+
)'loc:@bidirectional/backward_lstm_2/bias*
dtype0
Ķ
6bidirectional/backward_lstm_2/bias/Read/ReadVariableOpReadVariableOp"bidirectional/backward_lstm_2/bias*
_output_shapes	
:°	*5
_class+
)'loc:@bidirectional/backward_lstm_2/bias*
dtype0
g
bidirectional/ShapeShapedropout_2/cond/Merge*
T0*
out_type0*
_output_shapes
:
k
!bidirectional/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
m
#bidirectional/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
m
#bidirectional/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
æ
bidirectional/strided_sliceStridedSlicebidirectional/Shape!bidirectional/strided_slice/stack#bidirectional/strided_slice/stack_1#bidirectional/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
\
bidirectional/zeros/mul/yConst*
value
B :¬*
dtype0*
_output_shapes
: 
w
bidirectional/zeros/mulMulbidirectional/strided_slicebidirectional/zeros/mul/y*
_output_shapes
: *
T0
]
bidirectional/zeros/Less/yConst*
value
B :č*
dtype0*
_output_shapes
: 
v
bidirectional/zeros/LessLessbidirectional/zeros/mulbidirectional/zeros/Less/y*
_output_shapes
: *
T0
_
bidirectional/zeros/packed/1Const*
value
B :¬*
dtype0*
_output_shapes
: 

bidirectional/zeros/packedPackbidirectional/strided_slicebidirectional/zeros/packed/1*
_output_shapes
:*
T0*

axis *
N
^
bidirectional/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/zerosFillbidirectional/zeros/packedbidirectional/zeros/Const*(
_output_shapes
:’’’’’’’’’¬*
T0*

index_type0
^
bidirectional/zeros_1/mul/yConst*
_output_shapes
: *
value
B :¬*
dtype0
{
bidirectional/zeros_1/mulMulbidirectional/strided_slicebidirectional/zeros_1/mul/y*
_output_shapes
: *
T0
_
bidirectional/zeros_1/Less/yConst*
_output_shapes
: *
value
B :č*
dtype0
|
bidirectional/zeros_1/LessLessbidirectional/zeros_1/mulbidirectional/zeros_1/Less/y*
T0*
_output_shapes
: 
a
bidirectional/zeros_1/packed/1Const*
value
B :¬*
dtype0*
_output_shapes
: 

bidirectional/zeros_1/packedPackbidirectional/strided_slicebidirectional/zeros_1/packed/1*
T0*

axis *
N*
_output_shapes
:
`
bidirectional/zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

bidirectional/zeros_1Fillbidirectional/zeros_1/packedbidirectional/zeros_1/Const*
T0*

index_type0*(
_output_shapes
:’’’’’’’’’¬
q
bidirectional/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
„
bidirectional/transpose	Transposedropout_2/cond/Mergebidirectional/transpose/perm*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’ō*
Tperm0*
T0
l
bidirectional/Shape_1Shapebidirectional/transpose*
_output_shapes
:*
T0*
out_type0
m
#bidirectional/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%bidirectional/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
o
%bidirectional/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
É
bidirectional/strided_slice_1StridedSlicebidirectional/Shape_1#bidirectional/strided_slice_1/stack%bidirectional/strided_slice_1/stack_1%bidirectional/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
m
#bidirectional/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%bidirectional/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%bidirectional/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ż
bidirectional/strided_slice_2StridedSlicebidirectional/transpose#bidirectional/strided_slice_2/stack%bidirectional/strided_slice_2/stack_1%bidirectional/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *(
_output_shapes
:’’’’’’’’’ō

bidirectional/ReadVariableOpReadVariableOp#bidirectional/forward_lstm_2/kernel*
dtype0* 
_output_shapes
:
ō°	
t
#bidirectional/strided_slice_3/stackConst*
_output_shapes
:*
valueB"        *
dtype0
v
%bidirectional/strided_slice_3/stack_1Const*
valueB"    ,  *
dtype0*
_output_shapes
:
v
%bidirectional/strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Ś
bidirectional/strided_slice_3StridedSlicebidirectional/ReadVariableOp#bidirectional/strided_slice_3/stack%bidirectional/strided_slice_3/stack_1%bidirectional/strided_slice_3/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask* 
_output_shapes
:
ō¬*
T0*
Index0
µ
bidirectional/MatMulMatMulbidirectional/strided_slice_2bidirectional/strided_slice_3*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 

bidirectional/ReadVariableOp_1ReadVariableOp#bidirectional/forward_lstm_2/kernel*
dtype0* 
_output_shapes
:
ō°	
t
#bidirectional/strided_slice_4/stackConst*
valueB"    ,  *
dtype0*
_output_shapes
:
v
%bidirectional/strided_slice_4/stack_1Const*
valueB"    X  *
dtype0*
_output_shapes
:
v
%bidirectional/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Ü
bidirectional/strided_slice_4StridedSlicebidirectional/ReadVariableOp_1#bidirectional/strided_slice_4/stack%bidirectional/strided_slice_4/stack_1%bidirectional/strided_slice_4/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask* 
_output_shapes
:
ō¬
·
bidirectional/MatMul_1MatMulbidirectional/strided_slice_2bidirectional/strided_slice_4*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 

bidirectional/ReadVariableOp_2ReadVariableOp#bidirectional/forward_lstm_2/kernel* 
_output_shapes
:
ō°	*
dtype0
t
#bidirectional/strided_slice_5/stackConst*
valueB"    X  *
dtype0*
_output_shapes
:
v
%bidirectional/strided_slice_5/stack_1Const*
_output_shapes
:*
valueB"      *
dtype0
v
%bidirectional/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Ü
bidirectional/strided_slice_5StridedSlicebidirectional/ReadVariableOp_2#bidirectional/strided_slice_5/stack%bidirectional/strided_slice_5/stack_1%bidirectional/strided_slice_5/stack_2* 
_output_shapes
:
ō¬*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
·
bidirectional/MatMul_2MatMulbidirectional/strided_slice_2bidirectional/strided_slice_5*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( *
T0

bidirectional/ReadVariableOp_3ReadVariableOp#bidirectional/forward_lstm_2/kernel*
dtype0* 
_output_shapes
:
ō°	
t
#bidirectional/strided_slice_6/stackConst*
valueB"      *
dtype0*
_output_shapes
:
v
%bidirectional/strided_slice_6/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0
v
%bidirectional/strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Ü
bidirectional/strided_slice_6StridedSlicebidirectional/ReadVariableOp_3#bidirectional/strided_slice_6/stack%bidirectional/strided_slice_6/stack_1%bidirectional/strided_slice_6/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask* 
_output_shapes
:
ō¬
·
bidirectional/MatMul_3MatMulbidirectional/strided_slice_2bidirectional/strided_slice_6*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 
}
bidirectional/ReadVariableOp_4ReadVariableOp!bidirectional/forward_lstm_2/bias*
dtype0*
_output_shapes	
:°	
m
#bidirectional/strided_slice_7/stackConst*
_output_shapes
:*
valueB: *
dtype0
p
%bidirectional/strided_slice_7/stack_1Const*
_output_shapes
:*
valueB:¬*
dtype0
o
%bidirectional/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
×
bidirectional/strided_slice_7StridedSlicebidirectional/ReadVariableOp_4#bidirectional/strided_slice_7/stack%bidirectional/strided_slice_7/stack_1%bidirectional/strided_slice_7/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes	
:¬*
T0*
Index0

bidirectional/BiasAddBiasAddbidirectional/MatMulbidirectional/strided_slice_7*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬*
T0
}
bidirectional/ReadVariableOp_5ReadVariableOp!bidirectional/forward_lstm_2/bias*
dtype0*
_output_shapes	
:°	
n
#bidirectional/strided_slice_8/stackConst*
valueB:¬*
dtype0*
_output_shapes
:
p
%bidirectional/strided_slice_8/stack_1Const*
valueB:Ų*
dtype0*
_output_shapes
:
o
%bidirectional/strided_slice_8/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
×
bidirectional/strided_slice_8StridedSlicebidirectional/ReadVariableOp_5#bidirectional/strided_slice_8/stack%bidirectional/strided_slice_8/stack_1%bidirectional/strided_slice_8/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes	
:¬
£
bidirectional/BiasAdd_1BiasAddbidirectional/MatMul_1bidirectional/strided_slice_8*(
_output_shapes
:’’’’’’’’’¬*
T0*
data_formatNHWC
}
bidirectional/ReadVariableOp_6ReadVariableOp!bidirectional/forward_lstm_2/bias*
dtype0*
_output_shapes	
:°	
n
#bidirectional/strided_slice_9/stackConst*
valueB:Ų*
dtype0*
_output_shapes
:
p
%bidirectional/strided_slice_9/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%bidirectional/strided_slice_9/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
×
bidirectional/strided_slice_9StridedSlicebidirectional/ReadVariableOp_6#bidirectional/strided_slice_9/stack%bidirectional/strided_slice_9/stack_1%bidirectional/strided_slice_9/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:¬*
T0*
Index0*
shrink_axis_mask 
£
bidirectional/BiasAdd_2BiasAddbidirectional/MatMul_2bidirectional/strided_slice_9*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬*
T0
}
bidirectional/ReadVariableOp_7ReadVariableOp!bidirectional/forward_lstm_2/bias*
dtype0*
_output_shapes	
:°	
o
$bidirectional/strided_slice_10/stackConst*
_output_shapes
:*
valueB:*
dtype0
p
&bidirectional/strided_slice_10/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_10/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ū
bidirectional/strided_slice_10StridedSlicebidirectional/ReadVariableOp_7$bidirectional/strided_slice_10/stack&bidirectional/strided_slice_10/stack_1&bidirectional/strided_slice_10/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:¬*
Index0*
T0
¤
bidirectional/BiasAdd_3BiasAddbidirectional/MatMul_3bidirectional/strided_slice_10*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/ReadVariableOp_8ReadVariableOp-bidirectional/forward_lstm_2/recurrent_kernel* 
_output_shapes
:
¬°	*
dtype0
u
$bidirectional/strided_slice_11/stackConst*
valueB"        *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_11/stack_1Const*
valueB"    ,  *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_11/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ą
bidirectional/strided_slice_11StridedSlicebidirectional/ReadVariableOp_8$bidirectional/strided_slice_11/stack&bidirectional/strided_slice_11/stack_1&bidirectional/strided_slice_11/stack_2*
new_axis_mask *
end_mask* 
_output_shapes
:
¬¬*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask 
®
bidirectional/MatMul_4MatMulbidirectional/zerosbidirectional/strided_slice_11*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( *
T0
z
bidirectional/addAddbidirectional/BiasAddbidirectional/MatMul_4*(
_output_shapes
:’’’’’’’’’¬*
T0
X
bidirectional/mul/xConst*
_output_shapes
: *
valueB
 *ĶĢL>*
dtype0
s
bidirectional/mulMulbidirectional/mul/xbidirectional/add*
T0*(
_output_shapes
:’’’’’’’’’¬
Z
bidirectional/add_1/yConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
w
bidirectional/add_1Addbidirectional/mulbidirectional/add_1/y*
T0*(
_output_shapes
:’’’’’’’’’¬
X
bidirectional/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
bidirectional/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

#bidirectional/clip_by_value/MinimumMinimumbidirectional/add_1bidirectional/Const_1*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/clip_by_valueMaximum#bidirectional/clip_by_value/Minimumbidirectional/Const*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/ReadVariableOp_9ReadVariableOp-bidirectional/forward_lstm_2/recurrent_kernel*
dtype0* 
_output_shapes
:
¬°	
u
$bidirectional/strided_slice_12/stackConst*
valueB"    ,  *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_12/stack_1Const*
valueB"    X  *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_12/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ą
bidirectional/strided_slice_12StridedSlicebidirectional/ReadVariableOp_9$bidirectional/strided_slice_12/stack&bidirectional/strided_slice_12/stack_1&bidirectional/strided_slice_12/stack_2*
new_axis_mask *
end_mask* 
_output_shapes
:
¬¬*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask
®
bidirectional/MatMul_5MatMulbidirectional/zerosbidirectional/strided_slice_12*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( *
T0
~
bidirectional/add_2Addbidirectional/BiasAdd_1bidirectional/MatMul_5*
T0*(
_output_shapes
:’’’’’’’’’¬
Z
bidirectional/mul_1/xConst*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 
y
bidirectional/mul_1Mulbidirectional/mul_1/xbidirectional/add_2*(
_output_shapes
:’’’’’’’’’¬*
T0
Z
bidirectional/add_3/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
y
bidirectional/add_3Addbidirectional/mul_1bidirectional/add_3/y*
T0*(
_output_shapes
:’’’’’’’’’¬
Z
bidirectional/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
bidirectional/Const_3Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

%bidirectional/clip_by_value_1/MinimumMinimumbidirectional/add_3bidirectional/Const_3*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/clip_by_value_1Maximum%bidirectional/clip_by_value_1/Minimumbidirectional/Const_2*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/mul_2Mulbidirectional/clip_by_value_1bidirectional/zeros_1*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/ReadVariableOp_10ReadVariableOp-bidirectional/forward_lstm_2/recurrent_kernel*
dtype0* 
_output_shapes
:
¬°	
u
$bidirectional/strided_slice_13/stackConst*
valueB"    X  *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_13/stack_1Const*
valueB"      *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_13/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
į
bidirectional/strided_slice_13StridedSlicebidirectional/ReadVariableOp_10$bidirectional/strided_slice_13/stack&bidirectional/strided_slice_13/stack_1&bidirectional/strided_slice_13/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask* 
_output_shapes
:
¬¬
®
bidirectional/MatMul_6MatMulbidirectional/zerosbidirectional/strided_slice_13*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 
~
bidirectional/add_4Addbidirectional/BiasAdd_2bidirectional/MatMul_6*
T0*(
_output_shapes
:’’’’’’’’’¬
b
bidirectional/TanhTanhbidirectional/add_4*
T0*(
_output_shapes
:’’’’’’’’’¬
~
bidirectional/mul_3Mulbidirectional/clip_by_valuebidirectional/Tanh*
T0*(
_output_shapes
:’’’’’’’’’¬
w
bidirectional/add_5Addbidirectional/mul_2bidirectional/mul_3*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/ReadVariableOp_11ReadVariableOp-bidirectional/forward_lstm_2/recurrent_kernel*
dtype0* 
_output_shapes
:
¬°	
u
$bidirectional/strided_slice_14/stackConst*
_output_shapes
:*
valueB"      *
dtype0
w
&bidirectional/strided_slice_14/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_14/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
į
bidirectional/strided_slice_14StridedSlicebidirectional/ReadVariableOp_11$bidirectional/strided_slice_14/stack&bidirectional/strided_slice_14/stack_1&bidirectional/strided_slice_14/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask* 
_output_shapes
:
¬¬*
Index0*
T0*
shrink_axis_mask 
®
bidirectional/MatMul_7MatMulbidirectional/zerosbidirectional/strided_slice_14*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 
~
bidirectional/add_6Addbidirectional/BiasAdd_3bidirectional/MatMul_7*
T0*(
_output_shapes
:’’’’’’’’’¬
Z
bidirectional/mul_4/xConst*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 
y
bidirectional/mul_4Mulbidirectional/mul_4/xbidirectional/add_6*
T0*(
_output_shapes
:’’’’’’’’’¬
Z
bidirectional/add_7/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
y
bidirectional/add_7Addbidirectional/mul_4bidirectional/add_7/y*(
_output_shapes
:’’’’’’’’’¬*
T0
Z
bidirectional/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
bidirectional/Const_5Const*
_output_shapes
: *
valueB
 *  ?*
dtype0

%bidirectional/clip_by_value_2/MinimumMinimumbidirectional/add_7bidirectional/Const_5*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/clip_by_value_2Maximum%bidirectional/clip_by_value_2/Minimumbidirectional/Const_4*(
_output_shapes
:’’’’’’’’’¬*
T0
d
bidirectional/Tanh_1Tanhbidirectional/add_5*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/mul_5Mulbidirectional/clip_by_value_2bidirectional/Tanh_1*(
_output_shapes
:’’’’’’’’’¬*
T0
ł
bidirectional/TensorArrayTensorArrayV3bidirectional/strided_slice_1* 
tensor_array_name	output_ta*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
ś
bidirectional/TensorArray_1TensorArrayV3bidirectional/strided_slice_1*
tensor_array_name
input_ta*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
}
&bidirectional/TensorArrayUnstack/ShapeShapebidirectional/transpose*
T0*
out_type0*
_output_shapes
:
~
4bidirectional/TensorArrayUnstack/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0

6bidirectional/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

6bidirectional/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

.bidirectional/TensorArrayUnstack/strided_sliceStridedSlice&bidirectional/TensorArrayUnstack/Shape4bidirectional/TensorArrayUnstack/strided_slice/stack6bidirectional/TensorArrayUnstack/strided_slice/stack_16bidirectional/TensorArrayUnstack/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask 
n
,bidirectional/TensorArrayUnstack/range/startConst*
_output_shapes
: *
value	B : *
dtype0
n
,bidirectional/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ģ
&bidirectional/TensorArrayUnstack/rangeRange,bidirectional/TensorArrayUnstack/range/start.bidirectional/TensorArrayUnstack/strided_slice,bidirectional/TensorArrayUnstack/range/delta*#
_output_shapes
:’’’’’’’’’*

Tidx0
Ŗ
Hbidirectional/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3bidirectional/TensorArray_1&bidirectional/TensorArrayUnstack/rangebidirectional/transposebidirectional/TensorArray_1:1*
T0**
_class 
loc:@bidirectional/transpose*
_output_shapes
: 
T
bidirectional/timeConst*
value	B : *
dtype0*
_output_shapes
: 
ø
bidirectional/while/EnterEnterbidirectional/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *1

frame_name#!bidirectional/while/while_context
Ć
bidirectional/while/Enter_1Enterbidirectional/TensorArray:1*
_output_shapes
: *1

frame_name#!bidirectional/while/while_context*
T0*
is_constant( *
parallel_iterations 
Ķ
bidirectional/while/Enter_2Enterbidirectional/zeros*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:’’’’’’’’’¬*1

frame_name#!bidirectional/while/while_context
Ļ
bidirectional/while/Enter_3Enterbidirectional/zeros_1*
is_constant( *
parallel_iterations *(
_output_shapes
:’’’’’’’’’¬*1

frame_name#!bidirectional/while/while_context*
T0

bidirectional/while/MergeMergebidirectional/while/Enter!bidirectional/while/NextIteration*
T0*
N*
_output_shapes
: : 

bidirectional/while/Merge_1Mergebidirectional/while/Enter_1#bidirectional/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
¤
bidirectional/while/Merge_2Mergebidirectional/while/Enter_2#bidirectional/while/NextIteration_2**
_output_shapes
:’’’’’’’’’¬: *
T0*
N
¤
bidirectional/while/Merge_3Mergebidirectional/while/Enter_3#bidirectional/while/NextIteration_3*
N**
_output_shapes
:’’’’’’’’’¬: *
T0
|
bidirectional/while/LessLessbidirectional/while/Mergebidirectional/while/Less/Enter*
T0*
_output_shapes
: 
Č
bidirectional/while/Less/EnterEnterbidirectional/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *1

frame_name#!bidirectional/while/while_context
Z
bidirectional/while/LoopCondLoopCondbidirectional/while/Less*
_output_shapes
: 
®
bidirectional/while/SwitchSwitchbidirectional/while/Mergebidirectional/while/LoopCond*
T0*,
_class"
 loc:@bidirectional/while/Merge*
_output_shapes
: : 
“
bidirectional/while/Switch_1Switchbidirectional/while/Merge_1bidirectional/while/LoopCond*.
_class$
" loc:@bidirectional/while/Merge_1*
_output_shapes
: : *
T0
Ų
bidirectional/while/Switch_2Switchbidirectional/while/Merge_2bidirectional/while/LoopCond*.
_class$
" loc:@bidirectional/while/Merge_2*<
_output_shapes*
(:’’’’’’’’’¬:’’’’’’’’’¬*
T0
Ų
bidirectional/while/Switch_3Switchbidirectional/while/Merge_3bidirectional/while/LoopCond*
T0*.
_class$
" loc:@bidirectional/while/Merge_3*<
_output_shapes*
(:’’’’’’’’’¬:’’’’’’’’’¬
g
bidirectional/while/IdentityIdentitybidirectional/while/Switch:1*
T0*
_output_shapes
: 
k
bidirectional/while/Identity_1Identitybidirectional/while/Switch_1:1*
T0*
_output_shapes
: 
}
bidirectional/while/Identity_2Identitybidirectional/while/Switch_2:1*(
_output_shapes
:’’’’’’’’’¬*
T0
}
bidirectional/while/Identity_3Identitybidirectional/while/Switch_3:1*
T0*(
_output_shapes
:’’’’’’’’’¬
ė
%bidirectional/while/TensorArrayReadV3TensorArrayReadV3+bidirectional/while/TensorArrayReadV3/Enterbidirectional/while/Identity-bidirectional/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:’’’’’’’’’ō
×
+bidirectional/while/TensorArrayReadV3/EnterEnterbidirectional/TensorArray_1*
_output_shapes
:*1

frame_name#!bidirectional/while/while_context*
T0*
is_constant(*
parallel_iterations 

-bidirectional/while/TensorArrayReadV3/Enter_1EnterHbidirectional/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *1

frame_name#!bidirectional/while/while_context
¬
"bidirectional/while/ReadVariableOpReadVariableOp(bidirectional/while/ReadVariableOp/Enter^bidirectional/while/Identity*
dtype0* 
_output_shapes
:
ō°	
Ų
(bidirectional/while/ReadVariableOp/EnterEnter#bidirectional/forward_lstm_2/kernel*
is_constant(*
parallel_iterations *
_output_shapes
: *1

frame_name#!bidirectional/while/while_context*
T0

'bidirectional/while/strided_slice/stackConst^bidirectional/while/Identity*
valueB"        *
dtype0*
_output_shapes
:

)bidirectional/while/strided_slice/stack_1Const^bidirectional/while/Identity*
valueB"    ,  *
dtype0*
_output_shapes
:

)bidirectional/while/strided_slice/stack_2Const^bidirectional/while/Identity*
_output_shapes
:*
valueB"      *
dtype0
š
!bidirectional/while/strided_sliceStridedSlice"bidirectional/while/ReadVariableOp'bidirectional/while/strided_slice/stack)bidirectional/while/strided_slice/stack_1)bidirectional/while/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask* 
_output_shapes
:
ō¬
Ē
bidirectional/while/MatMulMatMul%bidirectional/while/TensorArrayReadV3!bidirectional/while/strided_slice*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 
®
$bidirectional/while/ReadVariableOp_1ReadVariableOp(bidirectional/while/ReadVariableOp/Enter^bidirectional/while/Identity*
dtype0* 
_output_shapes
:
ō°	

)bidirectional/while/strided_slice_1/stackConst^bidirectional/while/Identity*
valueB"    ,  *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_1/stack_1Const^bidirectional/while/Identity*
valueB"    X  *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_1/stack_2Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
ś
#bidirectional/while/strided_slice_1StridedSlice$bidirectional/while/ReadVariableOp_1)bidirectional/while/strided_slice_1/stack+bidirectional/while/strided_slice_1/stack_1+bidirectional/while/strided_slice_1/stack_2*
new_axis_mask *
end_mask* 
_output_shapes
:
ō¬*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask
Ė
bidirectional/while/MatMul_1MatMul%bidirectional/while/TensorArrayReadV3#bidirectional/while/strided_slice_1*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 
®
$bidirectional/while/ReadVariableOp_2ReadVariableOp(bidirectional/while/ReadVariableOp/Enter^bidirectional/while/Identity*
dtype0* 
_output_shapes
:
ō°	

)bidirectional/while/strided_slice_2/stackConst^bidirectional/while/Identity*
valueB"    X  *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_2/stack_1Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_2/stack_2Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
ś
#bidirectional/while/strided_slice_2StridedSlice$bidirectional/while/ReadVariableOp_2)bidirectional/while/strided_slice_2/stack+bidirectional/while/strided_slice_2/stack_1+bidirectional/while/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask* 
_output_shapes
:
ō¬*
Index0*
T0
Ė
bidirectional/while/MatMul_2MatMul%bidirectional/while/TensorArrayReadV3#bidirectional/while/strided_slice_2*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 
®
$bidirectional/while/ReadVariableOp_3ReadVariableOp(bidirectional/while/ReadVariableOp/Enter^bidirectional/while/Identity* 
_output_shapes
:
ō°	*
dtype0

)bidirectional/while/strided_slice_3/stackConst^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_3/stack_1Const^bidirectional/while/Identity*
_output_shapes
:*
valueB"        *
dtype0

+bidirectional/while/strided_slice_3/stack_2Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
ś
#bidirectional/while/strided_slice_3StridedSlice$bidirectional/while/ReadVariableOp_3)bidirectional/while/strided_slice_3/stack+bidirectional/while/strided_slice_3/stack_1+bidirectional/while/strided_slice_3/stack_2*
new_axis_mask *
end_mask* 
_output_shapes
:
ō¬*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask
Ė
bidirectional/while/MatMul_3MatMul%bidirectional/while/TensorArrayReadV3#bidirectional/while/strided_slice_3*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( *
T0
«
$bidirectional/while/ReadVariableOp_4ReadVariableOp*bidirectional/while/ReadVariableOp_4/Enter^bidirectional/while/Identity*
_output_shapes	
:°	*
dtype0
Ų
*bidirectional/while/ReadVariableOp_4/EnterEnter!bidirectional/forward_lstm_2/bias*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *1

frame_name#!bidirectional/while/while_context

)bidirectional/while/strided_slice_4/stackConst^bidirectional/while/Identity*
valueB: *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_4/stack_1Const^bidirectional/while/Identity*
_output_shapes
:*
valueB:¬*
dtype0

+bidirectional/while/strided_slice_4/stack_2Const^bidirectional/while/Identity*
valueB:*
dtype0*
_output_shapes
:
õ
#bidirectional/while/strided_slice_4StridedSlice$bidirectional/while/ReadVariableOp_4)bidirectional/while/strided_slice_4/stack+bidirectional/while/strided_slice_4/stack_1+bidirectional/while/strided_slice_4/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes	
:¬
±
bidirectional/while/BiasAddBiasAddbidirectional/while/MatMul#bidirectional/while/strided_slice_4*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬*
T0
«
$bidirectional/while/ReadVariableOp_5ReadVariableOp*bidirectional/while/ReadVariableOp_4/Enter^bidirectional/while/Identity*
_output_shapes	
:°	*
dtype0

)bidirectional/while/strided_slice_5/stackConst^bidirectional/while/Identity*
valueB:¬*
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_5/stack_1Const^bidirectional/while/Identity*
_output_shapes
:*
valueB:Ų*
dtype0

+bidirectional/while/strided_slice_5/stack_2Const^bidirectional/while/Identity*
valueB:*
dtype0*
_output_shapes
:
õ
#bidirectional/while/strided_slice_5StridedSlice$bidirectional/while/ReadVariableOp_5)bidirectional/while/strided_slice_5/stack+bidirectional/while/strided_slice_5/stack_1+bidirectional/while/strided_slice_5/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:¬*
Index0*
T0*
shrink_axis_mask 
µ
bidirectional/while/BiasAdd_1BiasAddbidirectional/while/MatMul_1#bidirectional/while/strided_slice_5*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬*
T0
«
$bidirectional/while/ReadVariableOp_6ReadVariableOp*bidirectional/while/ReadVariableOp_4/Enter^bidirectional/while/Identity*
dtype0*
_output_shapes	
:°	

)bidirectional/while/strided_slice_6/stackConst^bidirectional/while/Identity*
valueB:Ų*
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_6/stack_1Const^bidirectional/while/Identity*
valueB:*
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_6/stack_2Const^bidirectional/while/Identity*
valueB:*
dtype0*
_output_shapes
:
õ
#bidirectional/while/strided_slice_6StridedSlice$bidirectional/while/ReadVariableOp_6)bidirectional/while/strided_slice_6/stack+bidirectional/while/strided_slice_6/stack_1+bidirectional/while/strided_slice_6/stack_2*
_output_shapes	
:¬*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
µ
bidirectional/while/BiasAdd_2BiasAddbidirectional/while/MatMul_2#bidirectional/while/strided_slice_6*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬
«
$bidirectional/while/ReadVariableOp_7ReadVariableOp*bidirectional/while/ReadVariableOp_4/Enter^bidirectional/while/Identity*
dtype0*
_output_shapes	
:°	

)bidirectional/while/strided_slice_7/stackConst^bidirectional/while/Identity*
valueB:*
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_7/stack_1Const^bidirectional/while/Identity*
_output_shapes
:*
valueB: *
dtype0

+bidirectional/while/strided_slice_7/stack_2Const^bidirectional/while/Identity*
_output_shapes
:*
valueB:*
dtype0
õ
#bidirectional/while/strided_slice_7StridedSlice$bidirectional/while/ReadVariableOp_7)bidirectional/while/strided_slice_7/stack+bidirectional/while/strided_slice_7/stack_1+bidirectional/while/strided_slice_7/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:¬*
Index0*
T0
µ
bidirectional/while/BiasAdd_3BiasAddbidirectional/while/MatMul_3#bidirectional/while/strided_slice_7*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬
°
$bidirectional/while/ReadVariableOp_8ReadVariableOp*bidirectional/while/ReadVariableOp_8/Enter^bidirectional/while/Identity*
dtype0* 
_output_shapes
:
¬°	
ä
*bidirectional/while/ReadVariableOp_8/EnterEnter-bidirectional/forward_lstm_2/recurrent_kernel*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *1

frame_name#!bidirectional/while/while_context

)bidirectional/while/strided_slice_8/stackConst^bidirectional/while/Identity*
valueB"        *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_8/stack_1Const^bidirectional/while/Identity*
valueB"    ,  *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_8/stack_2Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
ś
#bidirectional/while/strided_slice_8StridedSlice$bidirectional/while/ReadVariableOp_8)bidirectional/while/strided_slice_8/stack+bidirectional/while/strided_slice_8/stack_1+bidirectional/while/strided_slice_8/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask* 
_output_shapes
:
¬¬*
T0*
Index0
Ä
bidirectional/while/MatMul_4MatMulbidirectional/while/Identity_2#bidirectional/while/strided_slice_8*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 

bidirectional/while/addAddbidirectional/while/BiasAddbidirectional/while/MatMul_4*
T0*(
_output_shapes
:’’’’’’’’’¬
}
bidirectional/while/mul/xConst^bidirectional/while/Identity*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 

bidirectional/while/mulMulbidirectional/while/mul/xbidirectional/while/add*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/while/add_1/yConst^bidirectional/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 

bidirectional/while/add_1Addbidirectional/while/mulbidirectional/while/add_1/y*(
_output_shapes
:’’’’’’’’’¬*
T0
}
bidirectional/while/ConstConst^bidirectional/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/while/Const_1Const^bidirectional/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

)bidirectional/while/clip_by_value/MinimumMinimumbidirectional/while/add_1bidirectional/while/Const_1*
T0*(
_output_shapes
:’’’’’’’’’¬
„
!bidirectional/while/clip_by_valueMaximum)bidirectional/while/clip_by_value/Minimumbidirectional/while/Const*
T0*(
_output_shapes
:’’’’’’’’’¬
°
$bidirectional/while/ReadVariableOp_9ReadVariableOp*bidirectional/while/ReadVariableOp_8/Enter^bidirectional/while/Identity*
dtype0* 
_output_shapes
:
¬°	

)bidirectional/while/strided_slice_9/stackConst^bidirectional/while/Identity*
valueB"    ,  *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_9/stack_1Const^bidirectional/while/Identity*
valueB"    X  *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_9/stack_2Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
ś
#bidirectional/while/strided_slice_9StridedSlice$bidirectional/while/ReadVariableOp_9)bidirectional/while/strided_slice_9/stack+bidirectional/while/strided_slice_9/stack_1+bidirectional/while/strided_slice_9/stack_2* 
_output_shapes
:
¬¬*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
Ä
bidirectional/while/MatMul_5MatMulbidirectional/while/Identity_2#bidirectional/while/strided_slice_9*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( *
T0

bidirectional/while/add_2Addbidirectional/while/BiasAdd_1bidirectional/while/MatMul_5*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/while/mul_1/xConst^bidirectional/while/Identity*
_output_shapes
: *
valueB
 *ĶĢL>*
dtype0

bidirectional/while/mul_1Mulbidirectional/while/mul_1/xbidirectional/while/add_2*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/while/add_3/yConst^bidirectional/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 

bidirectional/while/add_3Addbidirectional/while/mul_1bidirectional/while/add_3/y*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/while/Const_2Const^bidirectional/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/while/Const_3Const^bidirectional/while/Identity*
_output_shapes
: *
valueB
 *  ?*
dtype0
”
+bidirectional/while/clip_by_value_1/MinimumMinimumbidirectional/while/add_3bidirectional/while/Const_3*(
_output_shapes
:’’’’’’’’’¬*
T0
«
#bidirectional/while/clip_by_value_1Maximum+bidirectional/while/clip_by_value_1/Minimumbidirectional/while/Const_2*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/while/mul_2Mul#bidirectional/while/clip_by_value_1bidirectional/while/Identity_3*
T0*(
_output_shapes
:’’’’’’’’’¬
±
%bidirectional/while/ReadVariableOp_10ReadVariableOp*bidirectional/while/ReadVariableOp_8/Enter^bidirectional/while/Identity* 
_output_shapes
:
¬°	*
dtype0

*bidirectional/while/strided_slice_10/stackConst^bidirectional/while/Identity*
valueB"    X  *
dtype0*
_output_shapes
:

,bidirectional/while/strided_slice_10/stack_1Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:

,bidirectional/while/strided_slice_10/stack_2Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
’
$bidirectional/while/strided_slice_10StridedSlice%bidirectional/while/ReadVariableOp_10*bidirectional/while/strided_slice_10/stack,bidirectional/while/strided_slice_10/stack_1,bidirectional/while/strided_slice_10/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask* 
_output_shapes
:
¬¬
Å
bidirectional/while/MatMul_6MatMulbidirectional/while/Identity_2$bidirectional/while/strided_slice_10*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 

bidirectional/while/add_4Addbidirectional/while/BiasAdd_2bidirectional/while/MatMul_6*(
_output_shapes
:’’’’’’’’’¬*
T0
n
bidirectional/while/TanhTanhbidirectional/while/add_4*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/while/mul_3Mul!bidirectional/while/clip_by_valuebidirectional/while/Tanh*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/while/add_5Addbidirectional/while/mul_2bidirectional/while/mul_3*
T0*(
_output_shapes
:’’’’’’’’’¬
±
%bidirectional/while/ReadVariableOp_11ReadVariableOp*bidirectional/while/ReadVariableOp_8/Enter^bidirectional/while/Identity*
dtype0* 
_output_shapes
:
¬°	

*bidirectional/while/strided_slice_11/stackConst^bidirectional/while/Identity*
_output_shapes
:*
valueB"      *
dtype0

,bidirectional/while/strided_slice_11/stack_1Const^bidirectional/while/Identity*
_output_shapes
:*
valueB"        *
dtype0

,bidirectional/while/strided_slice_11/stack_2Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
’
$bidirectional/while/strided_slice_11StridedSlice%bidirectional/while/ReadVariableOp_11*bidirectional/while/strided_slice_11/stack,bidirectional/while/strided_slice_11/stack_1,bidirectional/while/strided_slice_11/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask* 
_output_shapes
:
¬¬
Å
bidirectional/while/MatMul_7MatMulbidirectional/while/Identity_2$bidirectional/while/strided_slice_11*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( *
T0

bidirectional/while/add_6Addbidirectional/while/BiasAdd_3bidirectional/while/MatMul_7*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/while/mul_4/xConst^bidirectional/while/Identity*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 

bidirectional/while/mul_4Mulbidirectional/while/mul_4/xbidirectional/while/add_6*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/while/add_7/yConst^bidirectional/while/Identity*
_output_shapes
: *
valueB
 *   ?*
dtype0

bidirectional/while/add_7Addbidirectional/while/mul_4bidirectional/while/add_7/y*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/while/Const_4Const^bidirectional/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/while/Const_5Const^bidirectional/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
”
+bidirectional/while/clip_by_value_2/MinimumMinimumbidirectional/while/add_7bidirectional/while/Const_5*
T0*(
_output_shapes
:’’’’’’’’’¬
«
#bidirectional/while/clip_by_value_2Maximum+bidirectional/while/clip_by_value_2/Minimumbidirectional/while/Const_4*
T0*(
_output_shapes
:’’’’’’’’’¬
p
bidirectional/while/Tanh_1Tanhbidirectional/while/add_5*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/while/mul_5Mul#bidirectional/while/clip_by_value_2bidirectional/while/Tanh_1*(
_output_shapes
:’’’’’’’’’¬*
T0
“
7bidirectional/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3=bidirectional/while/TensorArrayWrite/TensorArrayWriteV3/Enterbidirectional/while/Identitybidirectional/while/mul_5bidirectional/while/Identity_1*
_output_shapes
: *
T0*,
_class"
 loc:@bidirectional/while/mul_5

=bidirectional/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterbidirectional/TensorArray*
T0*,
_class"
 loc:@bidirectional/while/mul_5*
parallel_iterations *
is_constant(*1

frame_name#!bidirectional/while/while_context*
_output_shapes
:
|
bidirectional/while/add_8/yConst^bidirectional/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
|
bidirectional/while/add_8Addbidirectional/while/Identitybidirectional/while/add_8/y*
T0*
_output_shapes
: 
n
!bidirectional/while/NextIterationNextIterationbidirectional/while/add_8*
T0*
_output_shapes
: 

#bidirectional/while/NextIteration_1NextIteration7bidirectional/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 

#bidirectional/while/NextIteration_2NextIterationbidirectional/while/mul_5*
T0*(
_output_shapes
:’’’’’’’’’¬

#bidirectional/while/NextIteration_3NextIterationbidirectional/while/add_5*(
_output_shapes
:’’’’’’’’’¬*
T0
]
bidirectional/while/ExitExitbidirectional/while/Switch*
_output_shapes
: *
T0
a
bidirectional/while/Exit_1Exitbidirectional/while/Switch_1*
_output_shapes
: *
T0
s
bidirectional/while/Exit_2Exitbidirectional/while/Switch_2*
T0*(
_output_shapes
:’’’’’’’’’¬
s
bidirectional/while/Exit_3Exitbidirectional/while/Switch_3*
T0*(
_output_shapes
:’’’’’’’’’¬
Ā
0bidirectional/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3bidirectional/TensorArraybidirectional/while/Exit_1*,
_class"
 loc:@bidirectional/TensorArray*
_output_shapes
: 

*bidirectional/TensorArrayStack/range/startConst*,
_class"
 loc:@bidirectional/TensorArray*
value	B : *
dtype0*
_output_shapes
: 

*bidirectional/TensorArrayStack/range/deltaConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@bidirectional/TensorArray*
value	B :

$bidirectional/TensorArrayStack/rangeRange*bidirectional/TensorArrayStack/range/start0bidirectional/TensorArrayStack/TensorArraySizeV3*bidirectional/TensorArrayStack/range/delta*,
_class"
 loc:@bidirectional/TensorArray*#
_output_shapes
:’’’’’’’’’*

Tidx0
æ
2bidirectional/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3bidirectional/TensorArray$bidirectional/TensorArrayStack/rangebidirectional/while/Exit_1*%
element_shape:’’’’’’’’’¬*,
_class"
 loc:@bidirectional/TensorArray*
dtype0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’¬
U
bidirectional/sub/yConst*
_output_shapes
: *
value	B :*
dtype0
h
bidirectional/subSubbidirectional/while/Exitbidirectional/sub/y*
_output_shapes
: *
T0
µ
bidirectional/TensorArrayReadV3TensorArrayReadV3bidirectional/TensorArraybidirectional/subbidirectional/while/Exit_1*
dtype0*(
_output_shapes
:’’’’’’’’’¬
s
bidirectional/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
Ē
bidirectional/transpose_1	Transpose2bidirectional/TensorArrayStack/TensorArrayGatherV3bidirectional/transpose_1/perm*
Tperm0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’¬
i
bidirectional/Shape_2Shapedropout_2/cond/Merge*
_output_shapes
:*
T0*
out_type0
n
$bidirectional/strided_slice_15/stackConst*
valueB: *
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_15/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
p
&bidirectional/strided_slice_15/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ķ
bidirectional/strided_slice_15StridedSlicebidirectional/Shape_2$bidirectional/strided_slice_15/stack&bidirectional/strided_slice_15/stack_1&bidirectional/strided_slice_15/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
^
bidirectional/zeros_2/mul/yConst*
dtype0*
_output_shapes
: *
value
B :¬
~
bidirectional/zeros_2/mulMulbidirectional/strided_slice_15bidirectional/zeros_2/mul/y*
T0*
_output_shapes
: 
_
bidirectional/zeros_2/Less/yConst*
value
B :č*
dtype0*
_output_shapes
: 
|
bidirectional/zeros_2/LessLessbidirectional/zeros_2/mulbidirectional/zeros_2/Less/y*
T0*
_output_shapes
: 
a
bidirectional/zeros_2/packed/1Const*
value
B :¬*
dtype0*
_output_shapes
: 

bidirectional/zeros_2/packedPackbidirectional/strided_slice_15bidirectional/zeros_2/packed/1*
T0*

axis *
N*
_output_shapes
:
`
bidirectional/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/zeros_2Fillbidirectional/zeros_2/packedbidirectional/zeros_2/Const*
T0*

index_type0*(
_output_shapes
:’’’’’’’’’¬
^
bidirectional/zeros_3/mul/yConst*
value
B :¬*
dtype0*
_output_shapes
: 
~
bidirectional/zeros_3/mulMulbidirectional/strided_slice_15bidirectional/zeros_3/mul/y*
T0*
_output_shapes
: 
_
bidirectional/zeros_3/Less/yConst*
dtype0*
_output_shapes
: *
value
B :č
|
bidirectional/zeros_3/LessLessbidirectional/zeros_3/mulbidirectional/zeros_3/Less/y*
T0*
_output_shapes
: 
a
bidirectional/zeros_3/packed/1Const*
value
B :¬*
dtype0*
_output_shapes
: 

bidirectional/zeros_3/packedPackbidirectional/strided_slice_15bidirectional/zeros_3/packed/1*
_output_shapes
:*
T0*

axis *
N
`
bidirectional/zeros_3/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/zeros_3Fillbidirectional/zeros_3/packedbidirectional/zeros_3/Const*(
_output_shapes
:’’’’’’’’’¬*
T0*

index_type0
s
bidirectional/transpose_2/permConst*!
valueB"          *
dtype0*
_output_shapes
:
©
bidirectional/transpose_2	Transposedropout_2/cond/Mergebidirectional/transpose_2/perm*
Tperm0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’ō
f
bidirectional/ReverseV2/axisConst*
valueB: *
dtype0*
_output_shapes
:
©
bidirectional/ReverseV2	ReverseV2bidirectional/transpose_2bidirectional/ReverseV2/axis*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’ō*

Tidx0*
T0
l
bidirectional/Shape_3Shapebidirectional/ReverseV2*
T0*
out_type0*
_output_shapes
:
n
$bidirectional/strided_slice_16/stackConst*
valueB: *
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_16/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_16/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ķ
bidirectional/strided_slice_16StridedSlicebidirectional/Shape_3$bidirectional/strided_slice_16/stack&bidirectional/strided_slice_16/stack_1&bidirectional/strided_slice_16/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
n
$bidirectional/strided_slice_17/stackConst*
dtype0*
_output_shapes
:*
valueB: 
p
&bidirectional/strided_slice_17/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_17/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
į
bidirectional/strided_slice_17StridedSlicebidirectional/ReverseV2$bidirectional/strided_slice_17/stack&bidirectional/strided_slice_17/stack_1&bidirectional/strided_slice_17/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *(
_output_shapes
:’’’’’’’’’ō

bidirectional/ReadVariableOp_12ReadVariableOp$bidirectional/backward_lstm_2/kernel* 
_output_shapes
:
ō°	*
dtype0
u
$bidirectional/strided_slice_18/stackConst*
valueB"        *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_18/stack_1Const*
valueB"    ,  *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_18/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
į
bidirectional/strided_slice_18StridedSlicebidirectional/ReadVariableOp_12$bidirectional/strided_slice_18/stack&bidirectional/strided_slice_18/stack_1&bidirectional/strided_slice_18/stack_2* 
_output_shapes
:
ō¬*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
¹
bidirectional/MatMul_8MatMulbidirectional/strided_slice_17bidirectional/strided_slice_18*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 

bidirectional/ReadVariableOp_13ReadVariableOp$bidirectional/backward_lstm_2/kernel*
dtype0* 
_output_shapes
:
ō°	
u
$bidirectional/strided_slice_19/stackConst*
valueB"    ,  *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_19/stack_1Const*
valueB"    X  *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_19/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
į
bidirectional/strided_slice_19StridedSlicebidirectional/ReadVariableOp_13$bidirectional/strided_slice_19/stack&bidirectional/strided_slice_19/stack_1&bidirectional/strided_slice_19/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask* 
_output_shapes
:
ō¬
¹
bidirectional/MatMul_9MatMulbidirectional/strided_slice_17bidirectional/strided_slice_19*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( *
T0

bidirectional/ReadVariableOp_14ReadVariableOp$bidirectional/backward_lstm_2/kernel* 
_output_shapes
:
ō°	*
dtype0
u
$bidirectional/strided_slice_20/stackConst*
dtype0*
_output_shapes
:*
valueB"    X  
w
&bidirectional/strided_slice_20/stack_1Const*
_output_shapes
:*
valueB"      *
dtype0
w
&bidirectional/strided_slice_20/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
į
bidirectional/strided_slice_20StridedSlicebidirectional/ReadVariableOp_14$bidirectional/strided_slice_20/stack&bidirectional/strided_slice_20/stack_1&bidirectional/strided_slice_20/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask* 
_output_shapes
:
ō¬
ŗ
bidirectional/MatMul_10MatMulbidirectional/strided_slice_17bidirectional/strided_slice_20*
transpose_b( *
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( 

bidirectional/ReadVariableOp_15ReadVariableOp$bidirectional/backward_lstm_2/kernel* 
_output_shapes
:
ō°	*
dtype0
u
$bidirectional/strided_slice_21/stackConst*
valueB"      *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_21/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_21/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
į
bidirectional/strided_slice_21StridedSlicebidirectional/ReadVariableOp_15$bidirectional/strided_slice_21/stack&bidirectional/strided_slice_21/stack_1&bidirectional/strided_slice_21/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask* 
_output_shapes
:
ō¬
ŗ
bidirectional/MatMul_11MatMulbidirectional/strided_slice_17bidirectional/strided_slice_21*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 

bidirectional/ReadVariableOp_16ReadVariableOp"bidirectional/backward_lstm_2/bias*
_output_shapes	
:°	*
dtype0
n
$bidirectional/strided_slice_22/stackConst*
valueB: *
dtype0*
_output_shapes
:
q
&bidirectional/strided_slice_22/stack_1Const*
dtype0*
_output_shapes
:*
valueB:¬
p
&bidirectional/strided_slice_22/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ü
bidirectional/strided_slice_22StridedSlicebidirectional/ReadVariableOp_16$bidirectional/strided_slice_22/stack&bidirectional/strided_slice_22/stack_1&bidirectional/strided_slice_22/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:¬*
Index0*
T0
¤
bidirectional/BiasAdd_4BiasAddbidirectional/MatMul_8bidirectional/strided_slice_22*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬

bidirectional/ReadVariableOp_17ReadVariableOp"bidirectional/backward_lstm_2/bias*
dtype0*
_output_shapes	
:°	
o
$bidirectional/strided_slice_23/stackConst*
valueB:¬*
dtype0*
_output_shapes
:
q
&bidirectional/strided_slice_23/stack_1Const*
dtype0*
_output_shapes
:*
valueB:Ų
p
&bidirectional/strided_slice_23/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ü
bidirectional/strided_slice_23StridedSlicebidirectional/ReadVariableOp_17$bidirectional/strided_slice_23/stack&bidirectional/strided_slice_23/stack_1&bidirectional/strided_slice_23/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes	
:¬
¤
bidirectional/BiasAdd_5BiasAddbidirectional/MatMul_9bidirectional/strided_slice_23*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/ReadVariableOp_18ReadVariableOp"bidirectional/backward_lstm_2/bias*
dtype0*
_output_shapes	
:°	
o
$bidirectional/strided_slice_24/stackConst*
valueB:Ų*
dtype0*
_output_shapes
:
q
&bidirectional/strided_slice_24/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
p
&bidirectional/strided_slice_24/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ü
bidirectional/strided_slice_24StridedSlicebidirectional/ReadVariableOp_18$bidirectional/strided_slice_24/stack&bidirectional/strided_slice_24/stack_1&bidirectional/strided_slice_24/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes	
:¬*
T0*
Index0
„
bidirectional/BiasAdd_6BiasAddbidirectional/MatMul_10bidirectional/strided_slice_24*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬

bidirectional/ReadVariableOp_19ReadVariableOp"bidirectional/backward_lstm_2/bias*
_output_shapes	
:°	*
dtype0
o
$bidirectional/strided_slice_25/stackConst*
valueB:*
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_25/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_25/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ü
bidirectional/strided_slice_25StridedSlicebidirectional/ReadVariableOp_19$bidirectional/strided_slice_25/stack&bidirectional/strided_slice_25/stack_1&bidirectional/strided_slice_25/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes	
:¬*
T0*
Index0*
shrink_axis_mask 
„
bidirectional/BiasAdd_7BiasAddbidirectional/MatMul_11bidirectional/strided_slice_25*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/ReadVariableOp_20ReadVariableOp.bidirectional/backward_lstm_2/recurrent_kernel*
dtype0* 
_output_shapes
:
¬°	
u
$bidirectional/strided_slice_26/stackConst*
valueB"        *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_26/stack_1Const*
valueB"    ,  *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_26/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
į
bidirectional/strided_slice_26StridedSlicebidirectional/ReadVariableOp_20$bidirectional/strided_slice_26/stack&bidirectional/strided_slice_26/stack_1&bidirectional/strided_slice_26/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask* 
_output_shapes
:
¬¬
±
bidirectional/MatMul_12MatMulbidirectional/zeros_2bidirectional/strided_slice_26*
transpose_b( *
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( 

bidirectional/add_8Addbidirectional/BiasAdd_4bidirectional/MatMul_12*(
_output_shapes
:’’’’’’’’’¬*
T0
Z
bidirectional/mul_6/xConst*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 
y
bidirectional/mul_6Mulbidirectional/mul_6/xbidirectional/add_8*
T0*(
_output_shapes
:’’’’’’’’’¬
Z
bidirectional/add_9/yConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
y
bidirectional/add_9Addbidirectional/mul_6bidirectional/add_9/y*(
_output_shapes
:’’’’’’’’’¬*
T0
Z
bidirectional/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
bidirectional/Const_7Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

%bidirectional/clip_by_value_3/MinimumMinimumbidirectional/add_9bidirectional/Const_7*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/clip_by_value_3Maximum%bidirectional/clip_by_value_3/Minimumbidirectional/Const_6*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/ReadVariableOp_21ReadVariableOp.bidirectional/backward_lstm_2/recurrent_kernel*
dtype0* 
_output_shapes
:
¬°	
u
$bidirectional/strided_slice_27/stackConst*
valueB"    ,  *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_27/stack_1Const*
valueB"    X  *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_27/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
į
bidirectional/strided_slice_27StridedSlicebidirectional/ReadVariableOp_21$bidirectional/strided_slice_27/stack&bidirectional/strided_slice_27/stack_1&bidirectional/strided_slice_27/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask* 
_output_shapes
:
¬¬*
Index0*
T0
±
bidirectional/MatMul_13MatMulbidirectional/zeros_2bidirectional/strided_slice_27*
transpose_b( *
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( 

bidirectional/add_10Addbidirectional/BiasAdd_5bidirectional/MatMul_13*
T0*(
_output_shapes
:’’’’’’’’’¬
Z
bidirectional/mul_7/xConst*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 
z
bidirectional/mul_7Mulbidirectional/mul_7/xbidirectional/add_10*
T0*(
_output_shapes
:’’’’’’’’’¬
[
bidirectional/add_11/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
{
bidirectional/add_11Addbidirectional/mul_7bidirectional/add_11/y*
T0*(
_output_shapes
:’’’’’’’’’¬
Z
bidirectional/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
bidirectional/Const_9Const*
_output_shapes
: *
valueB
 *  ?*
dtype0

%bidirectional/clip_by_value_4/MinimumMinimumbidirectional/add_11bidirectional/Const_9*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/clip_by_value_4Maximum%bidirectional/clip_by_value_4/Minimumbidirectional/Const_8*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/mul_8Mulbidirectional/clip_by_value_4bidirectional/zeros_3*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/ReadVariableOp_22ReadVariableOp.bidirectional/backward_lstm_2/recurrent_kernel*
dtype0* 
_output_shapes
:
¬°	
u
$bidirectional/strided_slice_28/stackConst*
valueB"    X  *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_28/stack_1Const*
valueB"      *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_28/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
į
bidirectional/strided_slice_28StridedSlicebidirectional/ReadVariableOp_22$bidirectional/strided_slice_28/stack&bidirectional/strided_slice_28/stack_1&bidirectional/strided_slice_28/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask* 
_output_shapes
:
¬¬
±
bidirectional/MatMul_14MatMulbidirectional/zeros_2bidirectional/strided_slice_28*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 

bidirectional/add_12Addbidirectional/BiasAdd_6bidirectional/MatMul_14*
T0*(
_output_shapes
:’’’’’’’’’¬
e
bidirectional/Tanh_2Tanhbidirectional/add_12*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/mul_9Mulbidirectional/clip_by_value_3bidirectional/Tanh_2*(
_output_shapes
:’’’’’’’’’¬*
T0
x
bidirectional/add_13Addbidirectional/mul_8bidirectional/mul_9*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/ReadVariableOp_23ReadVariableOp.bidirectional/backward_lstm_2/recurrent_kernel*
dtype0* 
_output_shapes
:
¬°	
u
$bidirectional/strided_slice_29/stackConst*
valueB"      *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_29/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_29/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
į
bidirectional/strided_slice_29StridedSlicebidirectional/ReadVariableOp_23$bidirectional/strided_slice_29/stack&bidirectional/strided_slice_29/stack_1&bidirectional/strided_slice_29/stack_2*
end_mask* 
_output_shapes
:
¬¬*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask 
±
bidirectional/MatMul_15MatMulbidirectional/zeros_2bidirectional/strided_slice_29*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 

bidirectional/add_14Addbidirectional/BiasAdd_7bidirectional/MatMul_15*
T0*(
_output_shapes
:’’’’’’’’’¬
[
bidirectional/mul_10/xConst*
_output_shapes
: *
valueB
 *ĶĢL>*
dtype0
|
bidirectional/mul_10Mulbidirectional/mul_10/xbidirectional/add_14*
T0*(
_output_shapes
:’’’’’’’’’¬
[
bidirectional/add_15/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
|
bidirectional/add_15Addbidirectional/mul_10bidirectional/add_15/y*(
_output_shapes
:’’’’’’’’’¬*
T0
[
bidirectional/Const_10Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
bidirectional/Const_11Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

%bidirectional/clip_by_value_5/MinimumMinimumbidirectional/add_15bidirectional/Const_11*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/clip_by_value_5Maximum%bidirectional/clip_by_value_5/Minimumbidirectional/Const_10*(
_output_shapes
:’’’’’’’’’¬*
T0
e
bidirectional/Tanh_3Tanhbidirectional/add_13*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/mul_11Mulbidirectional/clip_by_value_5bidirectional/Tanh_3*
T0*(
_output_shapes
:’’’’’’’’’¬
ü
bidirectional/TensorArray_2TensorArrayV3bidirectional/strided_slice_16* 
tensor_array_name	output_ta*
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
ū
bidirectional/TensorArray_3TensorArrayV3bidirectional/strided_slice_16*
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*
tensor_array_name
input_ta

(bidirectional/TensorArrayUnstack_1/ShapeShapebidirectional/ReverseV2*
_output_shapes
:*
T0*
out_type0

6bidirectional/TensorArrayUnstack_1/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0

8bidirectional/TensorArrayUnstack_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

8bidirectional/TensorArrayUnstack_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ø
0bidirectional/TensorArrayUnstack_1/strided_sliceStridedSlice(bidirectional/TensorArrayUnstack_1/Shape6bidirectional/TensorArrayUnstack_1/strided_slice/stack8bidirectional/TensorArrayUnstack_1/strided_slice/stack_18bidirectional/TensorArrayUnstack_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
p
.bidirectional/TensorArrayUnstack_1/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
p
.bidirectional/TensorArrayUnstack_1/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ō
(bidirectional/TensorArrayUnstack_1/rangeRange.bidirectional/TensorArrayUnstack_1/range/start0bidirectional/TensorArrayUnstack_1/strided_slice.bidirectional/TensorArrayUnstack_1/range/delta*#
_output_shapes
:’’’’’’’’’*

Tidx0
®
Jbidirectional/TensorArrayUnstack_1/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3bidirectional/TensorArray_3(bidirectional/TensorArrayUnstack_1/rangebidirectional/ReverseV2bidirectional/TensorArray_3:1**
_class 
loc:@bidirectional/ReverseV2*
_output_shapes
: *
T0
V
bidirectional/time_1Const*
value	B : *
dtype0*
_output_shapes
: 
¾
bidirectional/while_1/EnterEnterbidirectional/time_1*
parallel_iterations *
_output_shapes
: *3

frame_name%#bidirectional/while_1/while_context*
T0*
is_constant( 
É
bidirectional/while_1/Enter_1Enterbidirectional/TensorArray_2:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *3

frame_name%#bidirectional/while_1/while_context
Ó
bidirectional/while_1/Enter_2Enterbidirectional/zeros_2*
parallel_iterations *(
_output_shapes
:’’’’’’’’’¬*3

frame_name%#bidirectional/while_1/while_context*
T0*
is_constant( 
Ó
bidirectional/while_1/Enter_3Enterbidirectional/zeros_3*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:’’’’’’’’’¬*3

frame_name%#bidirectional/while_1/while_context

bidirectional/while_1/MergeMergebidirectional/while_1/Enter#bidirectional/while_1/NextIteration*
T0*
N*
_output_shapes
: : 

bidirectional/while_1/Merge_1Mergebidirectional/while_1/Enter_1%bidirectional/while_1/NextIteration_1*
_output_shapes
: : *
T0*
N
Ŗ
bidirectional/while_1/Merge_2Mergebidirectional/while_1/Enter_2%bidirectional/while_1/NextIteration_2*
T0*
N**
_output_shapes
:’’’’’’’’’¬: 
Ŗ
bidirectional/while_1/Merge_3Mergebidirectional/while_1/Enter_3%bidirectional/while_1/NextIteration_3*
N**
_output_shapes
:’’’’’’’’’¬: *
T0

bidirectional/while_1/LessLessbidirectional/while_1/Merge bidirectional/while_1/Less/Enter*
_output_shapes
: *
T0
Ķ
 bidirectional/while_1/Less/EnterEnterbidirectional/strided_slice_16*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *3

frame_name%#bidirectional/while_1/while_context
^
bidirectional/while_1/LoopCondLoopCondbidirectional/while_1/Less*
_output_shapes
: 
¶
bidirectional/while_1/SwitchSwitchbidirectional/while_1/Mergebidirectional/while_1/LoopCond*
T0*.
_class$
" loc:@bidirectional/while_1/Merge*
_output_shapes
: : 
¼
bidirectional/while_1/Switch_1Switchbidirectional/while_1/Merge_1bidirectional/while_1/LoopCond*
T0*0
_class&
$"loc:@bidirectional/while_1/Merge_1*
_output_shapes
: : 
ą
bidirectional/while_1/Switch_2Switchbidirectional/while_1/Merge_2bidirectional/while_1/LoopCond*
T0*0
_class&
$"loc:@bidirectional/while_1/Merge_2*<
_output_shapes*
(:’’’’’’’’’¬:’’’’’’’’’¬
ą
bidirectional/while_1/Switch_3Switchbidirectional/while_1/Merge_3bidirectional/while_1/LoopCond*0
_class&
$"loc:@bidirectional/while_1/Merge_3*<
_output_shapes*
(:’’’’’’’’’¬:’’’’’’’’’¬*
T0
k
bidirectional/while_1/IdentityIdentitybidirectional/while_1/Switch:1*
T0*
_output_shapes
: 
o
 bidirectional/while_1/Identity_1Identity bidirectional/while_1/Switch_1:1*
_output_shapes
: *
T0

 bidirectional/while_1/Identity_2Identity bidirectional/while_1/Switch_2:1*(
_output_shapes
:’’’’’’’’’¬*
T0

 bidirectional/while_1/Identity_3Identity bidirectional/while_1/Switch_3:1*
T0*(
_output_shapes
:’’’’’’’’’¬
ó
'bidirectional/while_1/TensorArrayReadV3TensorArrayReadV3-bidirectional/while_1/TensorArrayReadV3/Enterbidirectional/while_1/Identity/bidirectional/while_1/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:’’’’’’’’’ō
Ū
-bidirectional/while_1/TensorArrayReadV3/EnterEnterbidirectional/TensorArray_3*
_output_shapes
:*3

frame_name%#bidirectional/while_1/while_context*
T0*
is_constant(*
parallel_iterations 

/bidirectional/while_1/TensorArrayReadV3/Enter_1EnterJbidirectional/TensorArrayUnstack_1/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *3

frame_name%#bidirectional/while_1/while_context
²
$bidirectional/while_1/ReadVariableOpReadVariableOp*bidirectional/while_1/ReadVariableOp/Enter^bidirectional/while_1/Identity*
dtype0* 
_output_shapes
:
ō°	
Ż
*bidirectional/while_1/ReadVariableOp/EnterEnter$bidirectional/backward_lstm_2/kernel*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *3

frame_name%#bidirectional/while_1/while_context

)bidirectional/while_1/strided_slice/stackConst^bidirectional/while_1/Identity*
_output_shapes
:*
valueB"        *
dtype0

+bidirectional/while_1/strided_slice/stack_1Const^bidirectional/while_1/Identity*
valueB"    ,  *
dtype0*
_output_shapes
:

+bidirectional/while_1/strided_slice/stack_2Const^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:
ś
#bidirectional/while_1/strided_sliceStridedSlice$bidirectional/while_1/ReadVariableOp)bidirectional/while_1/strided_slice/stack+bidirectional/while_1/strided_slice/stack_1+bidirectional/while_1/strided_slice/stack_2* 
_output_shapes
:
ō¬*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
Ķ
bidirectional/while_1/MatMulMatMul'bidirectional/while_1/TensorArrayReadV3#bidirectional/while_1/strided_slice*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 
“
&bidirectional/while_1/ReadVariableOp_1ReadVariableOp*bidirectional/while_1/ReadVariableOp/Enter^bidirectional/while_1/Identity*
dtype0* 
_output_shapes
:
ō°	

+bidirectional/while_1/strided_slice_1/stackConst^bidirectional/while_1/Identity*
valueB"    ,  *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_1/stack_1Const^bidirectional/while_1/Identity*
valueB"    X  *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_1/stack_2Const^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:

%bidirectional/while_1/strided_slice_1StridedSlice&bidirectional/while_1/ReadVariableOp_1+bidirectional/while_1/strided_slice_1/stack-bidirectional/while_1/strided_slice_1/stack_1-bidirectional/while_1/strided_slice_1/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask* 
_output_shapes
:
ō¬*
T0*
Index0
Ń
bidirectional/while_1/MatMul_1MatMul'bidirectional/while_1/TensorArrayReadV3%bidirectional/while_1/strided_slice_1*
transpose_b( *
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( 
“
&bidirectional/while_1/ReadVariableOp_2ReadVariableOp*bidirectional/while_1/ReadVariableOp/Enter^bidirectional/while_1/Identity*
dtype0* 
_output_shapes
:
ō°	

+bidirectional/while_1/strided_slice_2/stackConst^bidirectional/while_1/Identity*
valueB"    X  *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_2/stack_1Const^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_2/stack_2Const^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:

%bidirectional/while_1/strided_slice_2StridedSlice&bidirectional/while_1/ReadVariableOp_2+bidirectional/while_1/strided_slice_2/stack-bidirectional/while_1/strided_slice_2/stack_1-bidirectional/while_1/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask* 
_output_shapes
:
ō¬*
Index0*
T0
Ń
bidirectional/while_1/MatMul_2MatMul'bidirectional/while_1/TensorArrayReadV3%bidirectional/while_1/strided_slice_2*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( *
T0
“
&bidirectional/while_1/ReadVariableOp_3ReadVariableOp*bidirectional/while_1/ReadVariableOp/Enter^bidirectional/while_1/Identity*
dtype0* 
_output_shapes
:
ō°	

+bidirectional/while_1/strided_slice_3/stackConst^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_3/stack_1Const^bidirectional/while_1/Identity*
valueB"        *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_3/stack_2Const^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:

%bidirectional/while_1/strided_slice_3StridedSlice&bidirectional/while_1/ReadVariableOp_3+bidirectional/while_1/strided_slice_3/stack-bidirectional/while_1/strided_slice_3/stack_1-bidirectional/while_1/strided_slice_3/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask* 
_output_shapes
:
ō¬*
Index0*
T0
Ń
bidirectional/while_1/MatMul_3MatMul'bidirectional/while_1/TensorArrayReadV3%bidirectional/while_1/strided_slice_3*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( *
T0
±
&bidirectional/while_1/ReadVariableOp_4ReadVariableOp,bidirectional/while_1/ReadVariableOp_4/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes	
:°	
Ż
,bidirectional/while_1/ReadVariableOp_4/EnterEnter"bidirectional/backward_lstm_2/bias*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *3

frame_name%#bidirectional/while_1/while_context

+bidirectional/while_1/strided_slice_4/stackConst^bidirectional/while_1/Identity*
_output_shapes
:*
valueB: *
dtype0

-bidirectional/while_1/strided_slice_4/stack_1Const^bidirectional/while_1/Identity*
valueB:¬*
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_4/stack_2Const^bidirectional/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
’
%bidirectional/while_1/strided_slice_4StridedSlice&bidirectional/while_1/ReadVariableOp_4+bidirectional/while_1/strided_slice_4/stack-bidirectional/while_1/strided_slice_4/stack_1-bidirectional/while_1/strided_slice_4/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes	
:¬*
Index0*
T0
·
bidirectional/while_1/BiasAddBiasAddbidirectional/while_1/MatMul%bidirectional/while_1/strided_slice_4*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬
±
&bidirectional/while_1/ReadVariableOp_5ReadVariableOp,bidirectional/while_1/ReadVariableOp_4/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes	
:°	

+bidirectional/while_1/strided_slice_5/stackConst^bidirectional/while_1/Identity*
valueB:¬*
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_5/stack_1Const^bidirectional/while_1/Identity*
_output_shapes
:*
valueB:Ų*
dtype0

-bidirectional/while_1/strided_slice_5/stack_2Const^bidirectional/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
’
%bidirectional/while_1/strided_slice_5StridedSlice&bidirectional/while_1/ReadVariableOp_5+bidirectional/while_1/strided_slice_5/stack-bidirectional/while_1/strided_slice_5/stack_1-bidirectional/while_1/strided_slice_5/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes	
:¬
»
bidirectional/while_1/BiasAdd_1BiasAddbidirectional/while_1/MatMul_1%bidirectional/while_1/strided_slice_5*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬
±
&bidirectional/while_1/ReadVariableOp_6ReadVariableOp,bidirectional/while_1/ReadVariableOp_4/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes	
:°	

+bidirectional/while_1/strided_slice_6/stackConst^bidirectional/while_1/Identity*
valueB:Ų*
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_6/stack_1Const^bidirectional/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_6/stack_2Const^bidirectional/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
’
%bidirectional/while_1/strided_slice_6StridedSlice&bidirectional/while_1/ReadVariableOp_6+bidirectional/while_1/strided_slice_6/stack-bidirectional/while_1/strided_slice_6/stack_1-bidirectional/while_1/strided_slice_6/stack_2*
end_mask *
_output_shapes	
:¬*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask 
»
bidirectional/while_1/BiasAdd_2BiasAddbidirectional/while_1/MatMul_2%bidirectional/while_1/strided_slice_6*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬
±
&bidirectional/while_1/ReadVariableOp_7ReadVariableOp,bidirectional/while_1/ReadVariableOp_4/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes	
:°	

+bidirectional/while_1/strided_slice_7/stackConst^bidirectional/while_1/Identity*
_output_shapes
:*
valueB:*
dtype0

-bidirectional/while_1/strided_slice_7/stack_1Const^bidirectional/while_1/Identity*
valueB: *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_7/stack_2Const^bidirectional/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
’
%bidirectional/while_1/strided_slice_7StridedSlice&bidirectional/while_1/ReadVariableOp_7+bidirectional/while_1/strided_slice_7/stack-bidirectional/while_1/strided_slice_7/stack_1-bidirectional/while_1/strided_slice_7/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:¬*
T0*
Index0
»
bidirectional/while_1/BiasAdd_3BiasAddbidirectional/while_1/MatMul_3%bidirectional/while_1/strided_slice_7*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬
¶
&bidirectional/while_1/ReadVariableOp_8ReadVariableOp,bidirectional/while_1/ReadVariableOp_8/Enter^bidirectional/while_1/Identity*
dtype0* 
_output_shapes
:
¬°	
é
,bidirectional/while_1/ReadVariableOp_8/EnterEnter.bidirectional/backward_lstm_2/recurrent_kernel*
_output_shapes
: *3

frame_name%#bidirectional/while_1/while_context*
T0*
is_constant(*
parallel_iterations 

+bidirectional/while_1/strided_slice_8/stackConst^bidirectional/while_1/Identity*
valueB"        *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_8/stack_1Const^bidirectional/while_1/Identity*
valueB"    ,  *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_8/stack_2Const^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:

%bidirectional/while_1/strided_slice_8StridedSlice&bidirectional/while_1/ReadVariableOp_8+bidirectional/while_1/strided_slice_8/stack-bidirectional/while_1/strided_slice_8/stack_1-bidirectional/while_1/strided_slice_8/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask* 
_output_shapes
:
¬¬*
T0*
Index0
Ź
bidirectional/while_1/MatMul_4MatMul bidirectional/while_1/Identity_2%bidirectional/while_1/strided_slice_8*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 

bidirectional/while_1/addAddbidirectional/while_1/BiasAddbidirectional/while_1/MatMul_4*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/while_1/mul/xConst^bidirectional/while_1/Identity*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 

bidirectional/while_1/mulMulbidirectional/while_1/mul/xbidirectional/while_1/add*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/while_1/add_1/yConst^bidirectional/while_1/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 

bidirectional/while_1/add_1Addbidirectional/while_1/mulbidirectional/while_1/add_1/y*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/while_1/ConstConst^bidirectional/while_1/Identity*
_output_shapes
: *
valueB
 *    *
dtype0

bidirectional/while_1/Const_1Const^bidirectional/while_1/Identity*
_output_shapes
: *
valueB
 *  ?*
dtype0
„
+bidirectional/while_1/clip_by_value/MinimumMinimumbidirectional/while_1/add_1bidirectional/while_1/Const_1*(
_output_shapes
:’’’’’’’’’¬*
T0
«
#bidirectional/while_1/clip_by_valueMaximum+bidirectional/while_1/clip_by_value/Minimumbidirectional/while_1/Const*(
_output_shapes
:’’’’’’’’’¬*
T0
¶
&bidirectional/while_1/ReadVariableOp_9ReadVariableOp,bidirectional/while_1/ReadVariableOp_8/Enter^bidirectional/while_1/Identity*
dtype0* 
_output_shapes
:
¬°	

+bidirectional/while_1/strided_slice_9/stackConst^bidirectional/while_1/Identity*
valueB"    ,  *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_9/stack_1Const^bidirectional/while_1/Identity*
valueB"    X  *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_9/stack_2Const^bidirectional/while_1/Identity*
_output_shapes
:*
valueB"      *
dtype0

%bidirectional/while_1/strided_slice_9StridedSlice&bidirectional/while_1/ReadVariableOp_9+bidirectional/while_1/strided_slice_9/stack-bidirectional/while_1/strided_slice_9/stack_1-bidirectional/while_1/strided_slice_9/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask* 
_output_shapes
:
¬¬
Ź
bidirectional/while_1/MatMul_5MatMul bidirectional/while_1/Identity_2%bidirectional/while_1/strided_slice_9*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 

bidirectional/while_1/add_2Addbidirectional/while_1/BiasAdd_1bidirectional/while_1/MatMul_5*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/while_1/mul_1/xConst^bidirectional/while_1/Identity*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 

bidirectional/while_1/mul_1Mulbidirectional/while_1/mul_1/xbidirectional/while_1/add_2*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/while_1/add_3/yConst^bidirectional/while_1/Identity*
_output_shapes
: *
valueB
 *   ?*
dtype0

bidirectional/while_1/add_3Addbidirectional/while_1/mul_1bidirectional/while_1/add_3/y*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/while_1/Const_2Const^bidirectional/while_1/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/while_1/Const_3Const^bidirectional/while_1/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
§
-bidirectional/while_1/clip_by_value_1/MinimumMinimumbidirectional/while_1/add_3bidirectional/while_1/Const_3*(
_output_shapes
:’’’’’’’’’¬*
T0
±
%bidirectional/while_1/clip_by_value_1Maximum-bidirectional/while_1/clip_by_value_1/Minimumbidirectional/while_1/Const_2*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/while_1/mul_2Mul%bidirectional/while_1/clip_by_value_1 bidirectional/while_1/Identity_3*(
_output_shapes
:’’’’’’’’’¬*
T0
·
'bidirectional/while_1/ReadVariableOp_10ReadVariableOp,bidirectional/while_1/ReadVariableOp_8/Enter^bidirectional/while_1/Identity*
dtype0* 
_output_shapes
:
¬°	

,bidirectional/while_1/strided_slice_10/stackConst^bidirectional/while_1/Identity*
valueB"    X  *
dtype0*
_output_shapes
:
 
.bidirectional/while_1/strided_slice_10/stack_1Const^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:
 
.bidirectional/while_1/strided_slice_10/stack_2Const^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:

&bidirectional/while_1/strided_slice_10StridedSlice'bidirectional/while_1/ReadVariableOp_10,bidirectional/while_1/strided_slice_10/stack.bidirectional/while_1/strided_slice_10/stack_1.bidirectional/while_1/strided_slice_10/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask* 
_output_shapes
:
¬¬*
T0*
Index0
Ė
bidirectional/while_1/MatMul_6MatMul bidirectional/while_1/Identity_2&bidirectional/while_1/strided_slice_10*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 

bidirectional/while_1/add_4Addbidirectional/while_1/BiasAdd_2bidirectional/while_1/MatMul_6*
T0*(
_output_shapes
:’’’’’’’’’¬
r
bidirectional/while_1/TanhTanhbidirectional/while_1/add_4*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/while_1/mul_3Mul#bidirectional/while_1/clip_by_valuebidirectional/while_1/Tanh*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/while_1/add_5Addbidirectional/while_1/mul_2bidirectional/while_1/mul_3*(
_output_shapes
:’’’’’’’’’¬*
T0
·
'bidirectional/while_1/ReadVariableOp_11ReadVariableOp,bidirectional/while_1/ReadVariableOp_8/Enter^bidirectional/while_1/Identity*
dtype0* 
_output_shapes
:
¬°	

,bidirectional/while_1/strided_slice_11/stackConst^bidirectional/while_1/Identity*
_output_shapes
:*
valueB"      *
dtype0
 
.bidirectional/while_1/strided_slice_11/stack_1Const^bidirectional/while_1/Identity*
valueB"        *
dtype0*
_output_shapes
:
 
.bidirectional/while_1/strided_slice_11/stack_2Const^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:

&bidirectional/while_1/strided_slice_11StridedSlice'bidirectional/while_1/ReadVariableOp_11,bidirectional/while_1/strided_slice_11/stack.bidirectional/while_1/strided_slice_11/stack_1.bidirectional/while_1/strided_slice_11/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask* 
_output_shapes
:
¬¬*
T0*
Index0
Ė
bidirectional/while_1/MatMul_7MatMul bidirectional/while_1/Identity_2&bidirectional/while_1/strided_slice_11*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( *
T0

bidirectional/while_1/add_6Addbidirectional/while_1/BiasAdd_3bidirectional/while_1/MatMul_7*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/while_1/mul_4/xConst^bidirectional/while_1/Identity*
valueB
 *ĶĢL>*
dtype0*
_output_shapes
: 

bidirectional/while_1/mul_4Mulbidirectional/while_1/mul_4/xbidirectional/while_1/add_6*
T0*(
_output_shapes
:’’’’’’’’’¬

bidirectional/while_1/add_7/yConst^bidirectional/while_1/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 

bidirectional/while_1/add_7Addbidirectional/while_1/mul_4bidirectional/while_1/add_7/y*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/while_1/Const_4Const^bidirectional/while_1/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/while_1/Const_5Const^bidirectional/while_1/Identity*
_output_shapes
: *
valueB
 *  ?*
dtype0
§
-bidirectional/while_1/clip_by_value_2/MinimumMinimumbidirectional/while_1/add_7bidirectional/while_1/Const_5*
T0*(
_output_shapes
:’’’’’’’’’¬
±
%bidirectional/while_1/clip_by_value_2Maximum-bidirectional/while_1/clip_by_value_2/Minimumbidirectional/while_1/Const_4*
T0*(
_output_shapes
:’’’’’’’’’¬
t
bidirectional/while_1/Tanh_1Tanhbidirectional/while_1/add_5*(
_output_shapes
:’’’’’’’’’¬*
T0

bidirectional/while_1/mul_5Mul%bidirectional/while_1/clip_by_value_2bidirectional/while_1/Tanh_1*(
_output_shapes
:’’’’’’’’’¬*
T0
Ą
9bidirectional/while_1/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3?bidirectional/while_1/TensorArrayWrite/TensorArrayWriteV3/Enterbidirectional/while_1/Identitybidirectional/while_1/mul_5 bidirectional/while_1/Identity_1*.
_class$
" loc:@bidirectional/while_1/mul_5*
_output_shapes
: *
T0

?bidirectional/while_1/TensorArrayWrite/TensorArrayWriteV3/EnterEnterbidirectional/TensorArray_2*
is_constant(*3

frame_name%#bidirectional/while_1/while_context*
_output_shapes
:*
T0*.
_class$
" loc:@bidirectional/while_1/mul_5*
parallel_iterations 

bidirectional/while_1/add_8/yConst^bidirectional/while_1/Identity*
value	B :*
dtype0*
_output_shapes
: 

bidirectional/while_1/add_8Addbidirectional/while_1/Identitybidirectional/while_1/add_8/y*
T0*
_output_shapes
: 
r
#bidirectional/while_1/NextIterationNextIterationbidirectional/while_1/add_8*
T0*
_output_shapes
: 

%bidirectional/while_1/NextIteration_1NextIteration9bidirectional/while_1/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0

%bidirectional/while_1/NextIteration_2NextIterationbidirectional/while_1/mul_5*
T0*(
_output_shapes
:’’’’’’’’’¬

%bidirectional/while_1/NextIteration_3NextIterationbidirectional/while_1/add_5*(
_output_shapes
:’’’’’’’’’¬*
T0
a
bidirectional/while_1/ExitExitbidirectional/while_1/Switch*
_output_shapes
: *
T0
e
bidirectional/while_1/Exit_1Exitbidirectional/while_1/Switch_1*
_output_shapes
: *
T0
w
bidirectional/while_1/Exit_2Exitbidirectional/while_1/Switch_2*(
_output_shapes
:’’’’’’’’’¬*
T0
w
bidirectional/while_1/Exit_3Exitbidirectional/while_1/Switch_3*
T0*(
_output_shapes
:’’’’’’’’’¬
Ź
2bidirectional/TensorArrayStack_1/TensorArraySizeV3TensorArraySizeV3bidirectional/TensorArray_2bidirectional/while_1/Exit_1*.
_class$
" loc:@bidirectional/TensorArray_2*
_output_shapes
: 

,bidirectional/TensorArrayStack_1/range/startConst*.
_class$
" loc:@bidirectional/TensorArray_2*
value	B : *
dtype0*
_output_shapes
: 

,bidirectional/TensorArrayStack_1/range/deltaConst*
_output_shapes
: *.
_class$
" loc:@bidirectional/TensorArray_2*
value	B :*
dtype0
 
&bidirectional/TensorArrayStack_1/rangeRange,bidirectional/TensorArrayStack_1/range/start2bidirectional/TensorArrayStack_1/TensorArraySizeV3,bidirectional/TensorArrayStack_1/range/delta*.
_class$
" loc:@bidirectional/TensorArray_2*#
_output_shapes
:’’’’’’’’’*

Tidx0
É
4bidirectional/TensorArrayStack_1/TensorArrayGatherV3TensorArrayGatherV3bidirectional/TensorArray_2&bidirectional/TensorArrayStack_1/rangebidirectional/while_1/Exit_1*.
_class$
" loc:@bidirectional/TensorArray_2*
dtype0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’¬*%
element_shape:’’’’’’’’’¬
W
bidirectional/sub_1/yConst*
_output_shapes
: *
value	B :*
dtype0
n
bidirectional/sub_1Subbidirectional/while_1/Exitbidirectional/sub_1/y*
T0*
_output_shapes
: 
½
!bidirectional/TensorArrayReadV3_1TensorArrayReadV3bidirectional/TensorArray_2bidirectional/sub_1bidirectional/while_1/Exit_1*
dtype0*(
_output_shapes
:’’’’’’’’’¬
s
bidirectional/transpose_3/permConst*!
valueB"          *
dtype0*
_output_shapes
:
É
bidirectional/transpose_3	Transpose4bidirectional/TensorArrayStack_1/TensorArrayGatherV3bidirectional/transpose_3/perm*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’¬*
Tperm0
h
bidirectional/ReverseV2_1/axisConst*
valueB:*
dtype0*
_output_shapes
:
­
bidirectional/ReverseV2_1	ReverseV2bidirectional/transpose_3bidirectional/ReverseV2_1/axis*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’¬*

Tidx0*
T0
[
bidirectional/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Ę
bidirectional/concatConcatV2bidirectional/transpose_1bidirectional/ReverseV2_1bidirectional/concat/axis*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’Ų*

Tidx0*
T0*
N
~
dropout_3/cond/SwitchSwitchdropout/keras_learning_phasedropout/keras_learning_phase*
_output_shapes
: : *
T0

]
dropout_3/cond/switch_tIdentitydropout_3/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_3/cond/switch_fIdentitydropout_3/cond/Switch*
_output_shapes
: *
T0

a
dropout_3/cond/pred_idIdentitydropout/keras_learning_phase*
_output_shapes
: *
T0


 dropout_3/cond/dropout/keep_probConst^dropout_3/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 

dropout_3/cond/dropout/ShapeShape%dropout_3/cond/dropout/Shape/Switch:1*
out_type0*
_output_shapes
:*
T0
å
#dropout_3/cond/dropout/Shape/SwitchSwitchbidirectional/concatdropout_3/cond/pred_id*V
_output_shapesD
B:’’’’’’’’’’’’’’’’’’Ų:’’’’’’’’’’’’’’’’’’Ų*
T0*'
_class
loc:@bidirectional/concat

)dropout_3/cond/dropout/random_uniform/minConst^dropout_3/cond/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0

)dropout_3/cond/dropout/random_uniform/maxConst^dropout_3/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Č
3dropout_3/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_3/cond/dropout/Shape*
dtype0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’Ų*
seed2 *

seed *
T0
§
)dropout_3/cond/dropout/random_uniform/subSub)dropout_3/cond/dropout/random_uniform/max)dropout_3/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Š
)dropout_3/cond/dropout/random_uniform/mulMul3dropout_3/cond/dropout/random_uniform/RandomUniform)dropout_3/cond/dropout/random_uniform/sub*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’Ų
Ā
%dropout_3/cond/dropout/random_uniformAdd)dropout_3/cond/dropout/random_uniform/mul)dropout_3/cond/dropout/random_uniform/min*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’Ų*
T0
Ŗ
dropout_3/cond/dropout/addAdd dropout_3/cond/dropout/keep_prob%dropout_3/cond/dropout/random_uniform*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’Ų*
T0

dropout_3/cond/dropout/FloorFloordropout_3/cond/dropout/add*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’Ų*
T0
®
dropout_3/cond/dropout/divRealDiv%dropout_3/cond/dropout/Shape/Switch:1 dropout_3/cond/dropout/keep_prob*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’Ų

dropout_3/cond/dropout/mulMuldropout_3/cond/dropout/divdropout_3/cond/dropout/Floor*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’Ų

dropout_3/cond/IdentityIdentitydropout_3/cond/Identity/Switch*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’Ų
ą
dropout_3/cond/Identity/SwitchSwitchbidirectional/concatdropout_3/cond/pred_id*V
_output_shapesD
B:’’’’’’’’’’’’’’’’’’Ų:’’’’’’’’’’’’’’’’’’Ų*
T0*'
_class
loc:@bidirectional/concat

dropout_3/cond/MergeMergedropout_3/cond/Identitydropout_3/cond/dropout/mul*
N*7
_output_shapes%
#:’’’’’’’’’’’’’’’’’’Ų: *
T0

-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
valueB"X     *
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
_class
loc:@dense/kernel*
valueB
 *¢ĖÉ½*
dtype0

+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *¢ĖÉ=*
dtype0*
_output_shapes
: 
ę
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	Ų*

seed *
T0*
_class
loc:@dense/kernel*
seed2 
Ī
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
į
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	Ų*
T0*
_class
loc:@dense/kernel
Ó
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	Ų
§
dense/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense/kernel*
_class
loc:@dense/kernel*
	container *
shape:	Ų
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 

dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
dtype0

 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	Ų*
_class
loc:@dense/kernel*
dtype0

dense/bias/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB*    *
dtype0*
_output_shapes
:


dense/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_name
dense/bias*
_class
loc:@dense/bias*
	container *
shape:
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
{
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
_class
loc:@dense/bias*
dtype0

dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
l
dense/Tensordot/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	Ų
^
dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
e
dense/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
i
dense/Tensordot/ShapeShapedropout_3/cond/Merge*
_output_shapes
:*
T0*
out_type0
_
dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ø
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shapedense/Tensordot/freedense/Tensordot/GatherV2/axis*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0
a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
¼
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shapedense/Tensordot/axesdense/Tensordot/GatherV2_1/axis*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0
_
dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:

dense/Tensordot/ProdProddense/Tensordot/GatherV2dense/Tensordot/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
a
dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

dense/Tensordot/Prod_1Proddense/Tensordot/GatherV2_1dense/Tensordot/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
]
dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
„
dense/Tensordot/concatConcatV2dense/Tensordot/freedense/Tensordot/axesdense/Tensordot/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0

dense/Tensordot/stackPackdense/Tensordot/Proddense/Tensordot/Prod_1*

axis *
N*
_output_shapes
:*
T0
”
dense/Tensordot/transpose	Transposedropout_3/cond/Mergedense/Tensordot/concat*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’Ų*
Tperm0*
T0

dense/Tensordot/ReshapeReshapedense/Tensordot/transposedense/Tensordot/stack*
Tshape0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’*
T0
q
 dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
”
dense/Tensordot/transpose_1	Transposedense/Tensordot/ReadVariableOp dense/Tensordot/transpose_1/perm*
T0*
_output_shapes
:	Ų*
Tperm0
p
dense/Tensordot/Reshape_1/shapeConst*
valueB"X     *
dtype0*
_output_shapes
:

dense/Tensordot/Reshape_1Reshapedense/Tensordot/transpose_1dense/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	Ų
¬
dense/Tensordot/MatMulMatMuldense/Tensordot/Reshapedense/Tensordot/Reshape_1*'
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( *
T0
a
dense/Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
°
dense/Tensordot/concat_1ConcatV2dense/Tensordot/GatherV2dense/Tensordot/Const_2dense/Tensordot/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N

dense/TensordotReshapedense/Tensordot/MatMuldense/Tensordot/concat_1*
Tshape0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*
T0
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:

dense/BiasAddBiasAdddense/Tensordotdense/BiasAdd/ReadVariableOp*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*
T0*
data_formatNHWC
o
$predict_output/Max/reduction_indicesConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 
Ŗ
predict_output/MaxMaxdense/BiasAdd$predict_output/Max/reduction_indices*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*
	keep_dims(*

Tidx0*
T0
{
predict_output/subSubdense/BiasAddpredict_output/Max*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
l
predict_output/ExpExppredict_output/sub*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*
T0
o
$predict_output/Sum/reduction_indicesConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 
Æ
predict_output/SumSumpredict_output/Exp$predict_output/Sum/reduction_indices*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*
	keep_dims(*

Tidx0*
T0

predict_output/truedivRealDivpredict_output/Exppredict_output/Sum*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*
T0
\
PlaceholderPlaceholder*
dtype0*
_output_shapes

:fd*
shape
:fd
V
AssignVariableOpAssignVariableOpembedding_1/embeddingsPlaceholder*
dtype0
x
ReadVariableOpReadVariableOpembedding_1/embeddings^AssignVariableOp*
dtype0*
_output_shapes

:fd
`
Placeholder_1Placeholder*
dtype0*
_output_shapes
:	d*
shape:	d
O
AssignVariableOp_1AssignVariableOplstm/kernelPlaceholder_1*
dtype0
r
ReadVariableOp_1ReadVariableOplstm/kernel^AssignVariableOp_1*
_output_shapes
:	d*
dtype0
`
Placeholder_2Placeholder*
shape:	d*
dtype0*
_output_shapes
:	d
Y
AssignVariableOp_2AssignVariableOplstm/recurrent_kernelPlaceholder_2*
dtype0
|
ReadVariableOp_2ReadVariableOplstm/recurrent_kernel^AssignVariableOp_2*
_output_shapes
:	d*
dtype0
X
Placeholder_3Placeholder*
dtype0*
_output_shapes	
:*
shape:
M
AssignVariableOp_3AssignVariableOp	lstm/biasPlaceholder_3*
dtype0
l
ReadVariableOp_3ReadVariableOp	lstm/bias^AssignVariableOp_3*
_output_shapes	
:*
dtype0
`
Placeholder_4Placeholder*
dtype0*
_output_shapes
:	d*
shape:	d
Q
AssignVariableOp_4AssignVariableOplstm_1/kernelPlaceholder_4*
dtype0
t
ReadVariableOp_4ReadVariableOplstm_1/kernel^AssignVariableOp_4*
dtype0*
_output_shapes
:	d
`
Placeholder_5Placeholder*
_output_shapes
:	d*
shape:	d*
dtype0
[
AssignVariableOp_5AssignVariableOplstm_1/recurrent_kernelPlaceholder_5*
dtype0
~
ReadVariableOp_5ReadVariableOplstm_1/recurrent_kernel^AssignVariableOp_5*
_output_shapes
:	d*
dtype0
X
Placeholder_6Placeholder*
dtype0*
_output_shapes	
:*
shape:
O
AssignVariableOp_6AssignVariableOplstm_1/biasPlaceholder_6*
dtype0
n
ReadVariableOp_6ReadVariableOplstm_1/bias^AssignVariableOp_6*
_output_shapes	
:*
dtype0
d
Placeholder_7Placeholder*
dtype0*!
_output_shapes
:āō¬*
shape:āō¬
X
AssignVariableOp_7AssignVariableOpembedding/embeddingsPlaceholder_7*
dtype0
}
ReadVariableOp_7ReadVariableOpembedding/embeddings^AssignVariableOp_7*!
_output_shapes
:āō¬*
dtype0
b
Placeholder_8Placeholder*
dtype0* 
_output_shapes
:
ō°	*
shape:
ō°	
g
AssignVariableOp_8AssignVariableOp#bidirectional/forward_lstm_2/kernelPlaceholder_8*
dtype0

ReadVariableOp_8ReadVariableOp#bidirectional/forward_lstm_2/kernel^AssignVariableOp_8* 
_output_shapes
:
ō°	*
dtype0
b
Placeholder_9Placeholder*
shape:
¬°	*
dtype0* 
_output_shapes
:
¬°	
q
AssignVariableOp_9AssignVariableOp-bidirectional/forward_lstm_2/recurrent_kernelPlaceholder_9*
dtype0

ReadVariableOp_9ReadVariableOp-bidirectional/forward_lstm_2/recurrent_kernel^AssignVariableOp_9* 
_output_shapes
:
¬°	*
dtype0
Y
Placeholder_10Placeholder*
dtype0*
_output_shapes	
:°	*
shape:°	
g
AssignVariableOp_10AssignVariableOp!bidirectional/forward_lstm_2/biasPlaceholder_10*
dtype0

ReadVariableOp_10ReadVariableOp!bidirectional/forward_lstm_2/bias^AssignVariableOp_10*
_output_shapes	
:°	*
dtype0
c
Placeholder_11Placeholder*
dtype0* 
_output_shapes
:
ō°	*
shape:
ō°	
j
AssignVariableOp_11AssignVariableOp$bidirectional/backward_lstm_2/kernelPlaceholder_11*
dtype0

ReadVariableOp_11ReadVariableOp$bidirectional/backward_lstm_2/kernel^AssignVariableOp_11*
dtype0* 
_output_shapes
:
ō°	
c
Placeholder_12Placeholder*
dtype0* 
_output_shapes
:
¬°	*
shape:
¬°	
t
AssignVariableOp_12AssignVariableOp.bidirectional/backward_lstm_2/recurrent_kernelPlaceholder_12*
dtype0

ReadVariableOp_12ReadVariableOp.bidirectional/backward_lstm_2/recurrent_kernel^AssignVariableOp_12*
dtype0* 
_output_shapes
:
¬°	
Y
Placeholder_13Placeholder*
dtype0*
_output_shapes	
:°	*
shape:°	
h
AssignVariableOp_13AssignVariableOp"bidirectional/backward_lstm_2/biasPlaceholder_13*
dtype0

ReadVariableOp_13ReadVariableOp"bidirectional/backward_lstm_2/bias^AssignVariableOp_13*
dtype0*
_output_shapes	
:°	
a
Placeholder_14Placeholder*
_output_shapes
:	Ų*
shape:	Ų*
dtype0
R
AssignVariableOp_14AssignVariableOpdense/kernelPlaceholder_14*
dtype0
u
ReadVariableOp_14ReadVariableOpdense/kernel^AssignVariableOp_14*
dtype0*
_output_shapes
:	Ų
W
Placeholder_15Placeholder*
dtype0*
_output_shapes
:*
shape:
P
AssignVariableOp_15AssignVariableOp
dense/biasPlaceholder_15*
dtype0
n
ReadVariableOp_15ReadVariableOp
dense/bias^AssignVariableOp_15*
dtype0*
_output_shapes
:
e
VarIsInitializedOpVarIsInitializedOp#bidirectional/forward_lstm_2/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_1VarIsInitializedOplstm_1/kernel*
_output_shapes
: 
e
VarIsInitializedOp_2VarIsInitializedOp!bidirectional/forward_lstm_2/bias*
_output_shapes
: 
Z
VarIsInitializedOp_3VarIsInitializedOpembedding_1/embeddings*
_output_shapes
: 
P
VarIsInitializedOp_4VarIsInitializedOpdense/kernel*
_output_shapes
: 
[
VarIsInitializedOp_5VarIsInitializedOplstm_1/recurrent_kernel*
_output_shapes
: 
f
VarIsInitializedOp_6VarIsInitializedOp"bidirectional/backward_lstm_2/bias*
_output_shapes
: 
O
VarIsInitializedOp_7VarIsInitializedOplstm_1/bias*
_output_shapes
: 
q
VarIsInitializedOp_8VarIsInitializedOp-bidirectional/forward_lstm_2/recurrent_kernel*
_output_shapes
: 
X
VarIsInitializedOp_9VarIsInitializedOpembedding/embeddings*
_output_shapes
: 
i
VarIsInitializedOp_10VarIsInitializedOp$bidirectional/backward_lstm_2/kernel*
_output_shapes
: 
O
VarIsInitializedOp_11VarIsInitializedOp
dense/bias*
_output_shapes
: 
s
VarIsInitializedOp_12VarIsInitializedOp.bidirectional/backward_lstm_2/recurrent_kernel*
_output_shapes
: 
Z
VarIsInitializedOp_13VarIsInitializedOplstm/recurrent_kernel*
_output_shapes
: 
P
VarIsInitializedOp_14VarIsInitializedOplstm/kernel*
_output_shapes
: 
N
VarIsInitializedOp_15VarIsInitializedOp	lstm/bias*
_output_shapes
: 
©
initNoOp*^bidirectional/backward_lstm_2/bias/Assign,^bidirectional/backward_lstm_2/kernel/Assign6^bidirectional/backward_lstm_2/recurrent_kernel/Assign)^bidirectional/forward_lstm_2/bias/Assign+^bidirectional/forward_lstm_2/kernel/Assign5^bidirectional/forward_lstm_2/recurrent_kernel/Assign^dense/bias/Assign^dense/kernel/Assign^embedding/embeddings/Assign^embedding_1/embeddings/Assign^lstm/bias/Assign^lstm/kernel/Assign^lstm/recurrent_kernel/Assign^lstm_1/bias/Assign^lstm_1/kernel/Assign^lstm_1/recurrent_kernel/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_bd3ef52c512a4626a15d457607005de6/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
’
save/SaveV2/tensor_namesConst*²
valueØB„B"bidirectional/backward_lstm_2/biasB$bidirectional/backward_lstm_2/kernelB.bidirectional/backward_lstm_2/recurrent_kernelB!bidirectional/forward_lstm_2/biasB#bidirectional/forward_lstm_2/kernelB-bidirectional/forward_lstm_2/recurrent_kernelB
dense/biasBdense/kernelBembedding/embeddingsBembedding_1/embeddingsB	lstm/biasBlstm/kernelBlstm/recurrent_kernelBlstm_1/biasBlstm_1/kernelBlstm_1/recurrent_kernel*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ą
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices6bidirectional/backward_lstm_2/bias/Read/ReadVariableOp8bidirectional/backward_lstm_2/kernel/Read/ReadVariableOpBbidirectional/backward_lstm_2/recurrent_kernel/Read/ReadVariableOp5bidirectional/forward_lstm_2/bias/Read/ReadVariableOp7bidirectional/forward_lstm_2/kernel/Read/ReadVariableOpAbidirectional/forward_lstm_2/recurrent_kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp(embedding/embeddings/Read/ReadVariableOp*embedding_1/embeddings/Read/ReadVariableOplstm/bias/Read/ReadVariableOplstm/kernel/Read/ReadVariableOp)lstm/recurrent_kernel/Read/ReadVariableOplstm_1/bias/Read/ReadVariableOp!lstm_1/kernel/Read/ReadVariableOp+lstm_1/recurrent_kernel/Read/ReadVariableOp*
dtypes
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
_output_shapes
:*
T0*

axis *
N
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
_output_shapes
: *
T0

save/RestoreV2/tensor_namesConst*
_output_shapes
:*²
valueØB„B"bidirectional/backward_lstm_2/biasB$bidirectional/backward_lstm_2/kernelB.bidirectional/backward_lstm_2/recurrent_kernelB!bidirectional/forward_lstm_2/biasB#bidirectional/forward_lstm_2/kernelB-bidirectional/forward_lstm_2/recurrent_kernelB
dense/biasBdense/kernelBembedding/embeddingsBembedding_1/embeddingsB	lstm/biasBlstm/kernelBlstm/recurrent_kernelBlstm_1/biasBlstm_1/kernelBlstm_1/recurrent_kernel*
dtype0

save/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Ū
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*T
_output_shapesB
@::::::::::::::::*
dtypes
2
N
save/Identity_1Identitysave/RestoreV2*
_output_shapes
:*
T0
k
save/AssignVariableOpAssignVariableOp"bidirectional/backward_lstm_2/biassave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
o
save/AssignVariableOp_1AssignVariableOp$bidirectional/backward_lstm_2/kernelsave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
_output_shapes
:*
T0
y
save/AssignVariableOp_2AssignVariableOp.bidirectional/backward_lstm_2/recurrent_kernelsave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
_output_shapes
:*
T0
l
save/AssignVariableOp_3AssignVariableOp!bidirectional/forward_lstm_2/biassave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
n
save/AssignVariableOp_4AssignVariableOp#bidirectional/forward_lstm_2/kernelsave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
_output_shapes
:*
T0
x
save/AssignVariableOp_5AssignVariableOp-bidirectional/forward_lstm_2/recurrent_kernelsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
U
save/AssignVariableOp_6AssignVariableOp
dense/biassave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
_output_shapes
:*
T0
W
save/AssignVariableOp_7AssignVariableOpdense/kernelsave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
_
save/AssignVariableOp_8AssignVariableOpembedding/embeddingssave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
b
save/AssignVariableOp_9AssignVariableOpembedding_1/embeddingssave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
V
save/AssignVariableOp_10AssignVariableOp	lstm/biassave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
X
save/AssignVariableOp_11AssignVariableOplstm/kernelsave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0*
_output_shapes
:
b
save/AssignVariableOp_12AssignVariableOplstm/recurrent_kernelsave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
X
save/AssignVariableOp_13AssignVariableOplstm_1/biassave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
_output_shapes
:*
T0
Z
save/AssignVariableOp_14AssignVariableOplstm_1/kernelsave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
_output_shapes
:*
T0
d
save/AssignVariableOp_15AssignVariableOplstm_1/recurrent_kernelsave/Identity_16*
dtype0
¾
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"¶#
cond_context„#¢#

dropout/cond/cond_textdropout/cond/pred_id:0dropout/cond/switch_t:0 *æ
dropout/cond/dropout/Floor:0
#dropout/cond/dropout/Shape/Switch:1
dropout/cond/dropout/Shape:0
dropout/cond/dropout/add:0
dropout/cond/dropout/div:0
 dropout/cond/dropout/keep_prob:0
dropout/cond/dropout/mul:0
3dropout/cond/dropout/random_uniform/RandomUniform:0
)dropout/cond/dropout/random_uniform/max:0
)dropout/cond/dropout/random_uniform/min:0
)dropout/cond/dropout/random_uniform/mul:0
)dropout/cond/dropout/random_uniform/sub:0
%dropout/cond/dropout/random_uniform:0
dropout/cond/pred_id:0
dropout/cond/switch_t:0
)embedding_1/embedding_lookup/Identity_2:0P
)embedding_1/embedding_lookup/Identity_2:0#dropout/cond/dropout/Shape/Switch:10
dropout/cond/pred_id:0dropout/cond/pred_id:0
ā
dropout/cond/cond_text_1dropout/cond/pred_id:0dropout/cond/switch_f:0*
dropout/cond/Identity/Switch:0
dropout/cond/Identity:0
dropout/cond/pred_id:0
dropout/cond/switch_f:0
)embedding_1/embedding_lookup/Identity_2:0K
)embedding_1/embedding_lookup/Identity_2:0dropout/cond/Identity/Switch:00
dropout/cond/pred_id:0dropout/cond/pred_id:0

dropout_1/cond/cond_textdropout_1/cond/pred_id:0dropout_1/cond/switch_t:0 *¹
concatenate/concat:0
dropout_1/cond/dropout/Floor:0
%dropout_1/cond/dropout/Shape/Switch:1
dropout_1/cond/dropout/Shape:0
dropout_1/cond/dropout/add:0
dropout_1/cond/dropout/div:0
"dropout_1/cond/dropout/keep_prob:0
dropout_1/cond/dropout/mul:0
5dropout_1/cond/dropout/random_uniform/RandomUniform:0
+dropout_1/cond/dropout/random_uniform/max:0
+dropout_1/cond/dropout/random_uniform/min:0
+dropout_1/cond/dropout/random_uniform/mul:0
+dropout_1/cond/dropout/random_uniform/sub:0
'dropout_1/cond/dropout/random_uniform:0
dropout_1/cond/pred_id:0
dropout_1/cond/switch_t:04
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:0=
concatenate/concat:0%dropout_1/cond/dropout/Shape/Switch:1
Ģ
dropout_1/cond/cond_text_1dropout_1/cond/pred_id:0dropout_1/cond/switch_f:0*ų
concatenate/concat:0
 dropout_1/cond/Identity/Switch:0
dropout_1/cond/Identity:0
dropout_1/cond/pred_id:0
dropout_1/cond/switch_f:04
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:08
concatenate/concat:0 dropout_1/cond/Identity/Switch:0

dropout_2/cond/cond_textdropout_2/cond/pred_id:0dropout_2/cond/switch_t:0 *½
concatenate_1/concat:0
dropout_2/cond/dropout/Floor:0
%dropout_2/cond/dropout/Shape/Switch:1
dropout_2/cond/dropout/Shape:0
dropout_2/cond/dropout/add:0
dropout_2/cond/dropout/div:0
"dropout_2/cond/dropout/keep_prob:0
dropout_2/cond/dropout/mul:0
5dropout_2/cond/dropout/random_uniform/RandomUniform:0
+dropout_2/cond/dropout/random_uniform/max:0
+dropout_2/cond/dropout/random_uniform/min:0
+dropout_2/cond/dropout/random_uniform/mul:0
+dropout_2/cond/dropout/random_uniform/sub:0
'dropout_2/cond/dropout/random_uniform:0
dropout_2/cond/pred_id:0
dropout_2/cond/switch_t:04
dropout_2/cond/pred_id:0dropout_2/cond/pred_id:0?
concatenate_1/concat:0%dropout_2/cond/dropout/Shape/Switch:1
Š
dropout_2/cond/cond_text_1dropout_2/cond/pred_id:0dropout_2/cond/switch_f:0*ü
concatenate_1/concat:0
 dropout_2/cond/Identity/Switch:0
dropout_2/cond/Identity:0
dropout_2/cond/pred_id:0
dropout_2/cond/switch_f:0:
concatenate_1/concat:0 dropout_2/cond/Identity/Switch:04
dropout_2/cond/pred_id:0dropout_2/cond/pred_id:0

dropout_3/cond/cond_textdropout_3/cond/pred_id:0dropout_3/cond/switch_t:0 *½
bidirectional/concat:0
dropout_3/cond/dropout/Floor:0
%dropout_3/cond/dropout/Shape/Switch:1
dropout_3/cond/dropout/Shape:0
dropout_3/cond/dropout/add:0
dropout_3/cond/dropout/div:0
"dropout_3/cond/dropout/keep_prob:0
dropout_3/cond/dropout/mul:0
5dropout_3/cond/dropout/random_uniform/RandomUniform:0
+dropout_3/cond/dropout/random_uniform/max:0
+dropout_3/cond/dropout/random_uniform/min:0
+dropout_3/cond/dropout/random_uniform/mul:0
+dropout_3/cond/dropout/random_uniform/sub:0
'dropout_3/cond/dropout/random_uniform:0
dropout_3/cond/pred_id:0
dropout_3/cond/switch_t:04
dropout_3/cond/pred_id:0dropout_3/cond/pred_id:0?
bidirectional/concat:0%dropout_3/cond/dropout/Shape/Switch:1
Š
dropout_3/cond/cond_text_1dropout_3/cond/pred_id:0dropout_3/cond/switch_f:0*ü
bidirectional/concat:0
 dropout_3/cond/Identity/Switch:0
dropout_3/cond/Identity:0
dropout_3/cond/pred_id:0
dropout_3/cond/switch_f:04
dropout_3/cond/pred_id:0dropout_3/cond/pred_id:0:
bidirectional/concat:0 dropout_3/cond/Identity/Switch:0"ą
	variablesŅĻ

embedding/embeddings:0embedding/embeddings/Assign*embedding/embeddings/Read/ReadVariableOp:0(21embedding/embeddings/Initializer/random_uniform:0
 
embedding_1/embeddings:0embedding_1/embeddings/Assign,embedding_1/embeddings/Read/ReadVariableOp:0(23embedding_1/embeddings/Initializer/random_uniform:08
t
lstm/kernel:0lstm/kernel/Assign!lstm/kernel/Read/ReadVariableOp:0(2(lstm/kernel/Initializer/random_uniform:08

lstm/recurrent_kernel:0lstm/recurrent_kernel/Assign+lstm/recurrent_kernel/Read/ReadVariableOp:0(2)lstm/recurrent_kernel/Initializer/mul_1:08
d
lstm/bias:0lstm/bias/Assignlstm/bias/Read/ReadVariableOp:0(2lstm/bias/Initializer/concat:08
|
lstm_1/kernel:0lstm_1/kernel/Assign#lstm_1/kernel/Read/ReadVariableOp:0(2*lstm_1/kernel/Initializer/random_uniform:08

lstm_1/recurrent_kernel:0lstm_1/recurrent_kernel/Assign-lstm_1/recurrent_kernel/Read/ReadVariableOp:0(2+lstm_1/recurrent_kernel/Initializer/mul_1:08
l
lstm_1/bias:0lstm_1/bias/Assign!lstm_1/bias/Read/ReadVariableOp:0(2 lstm_1/bias/Initializer/concat:08
Ō
%bidirectional/forward_lstm_2/kernel:0*bidirectional/forward_lstm_2/kernel/Assign9bidirectional/forward_lstm_2/kernel/Read/ReadVariableOp:0(2@bidirectional/forward_lstm_2/kernel/Initializer/random_uniform:08
ó
/bidirectional/forward_lstm_2/recurrent_kernel:04bidirectional/forward_lstm_2/recurrent_kernel/AssignCbidirectional/forward_lstm_2/recurrent_kernel/Read/ReadVariableOp:0(2Abidirectional/forward_lstm_2/recurrent_kernel/Initializer/mul_1:08
Ä
#bidirectional/forward_lstm_2/bias:0(bidirectional/forward_lstm_2/bias/Assign7bidirectional/forward_lstm_2/bias/Read/ReadVariableOp:0(26bidirectional/forward_lstm_2/bias/Initializer/concat:08
Ų
&bidirectional/backward_lstm_2/kernel:0+bidirectional/backward_lstm_2/kernel/Assign:bidirectional/backward_lstm_2/kernel/Read/ReadVariableOp:0(2Abidirectional/backward_lstm_2/kernel/Initializer/random_uniform:08
÷
0bidirectional/backward_lstm_2/recurrent_kernel:05bidirectional/backward_lstm_2/recurrent_kernel/AssignDbidirectional/backward_lstm_2/recurrent_kernel/Read/ReadVariableOp:0(2Bbidirectional/backward_lstm_2/recurrent_kernel/Initializer/mul_1:08
Č
$bidirectional/backward_lstm_2/bias:0)bidirectional/backward_lstm_2/bias/Assign8bidirectional/backward_lstm_2/bias/Read/ReadVariableOp:0(27bidirectional/backward_lstm_2/bias/Initializer/concat:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08"°Ä
while_contextÄÄ
å(
lstm/while/while_context  *lstm/while/LoopCond:02lstm/while/Merge:0:lstm/while/Identity:0Blstm/while/Exit:0Blstm/while/Exit_1:0Blstm/while/Exit_2:0Blstm/while/Exit_3:0JŲ&
lstm/TensorArray:0
Alstm/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
lstm/TensorArray_1:0
lstm/bias:0
lstm/kernel:0
lstm/recurrent_kernel:0
lstm/strided_slice_1:0
lstm/while/BiasAdd:0
lstm/while/BiasAdd_1:0
lstm/while/BiasAdd_2:0
lstm/while/BiasAdd_3:0
lstm/while/Const:0
lstm/while/Const_1:0
lstm/while/Const_2:0
lstm/while/Const_3:0
lstm/while/Const_4:0
lstm/while/Const_5:0
lstm/while/Enter:0
lstm/while/Enter_1:0
lstm/while/Enter_2:0
lstm/while/Enter_3:0
lstm/while/Exit:0
lstm/while/Exit_1:0
lstm/while/Exit_2:0
lstm/while/Exit_3:0
lstm/while/Identity:0
lstm/while/Identity_1:0
lstm/while/Identity_2:0
lstm/while/Identity_3:0
lstm/while/Less/Enter:0
lstm/while/Less:0
lstm/while/LoopCond:0
lstm/while/MatMul:0
lstm/while/MatMul_1:0
lstm/while/MatMul_2:0
lstm/while/MatMul_3:0
lstm/while/MatMul_4:0
lstm/while/MatMul_5:0
lstm/while/MatMul_6:0
lstm/while/MatMul_7:0
lstm/while/Merge:0
lstm/while/Merge:1
lstm/while/Merge_1:0
lstm/while/Merge_1:1
lstm/while/Merge_2:0
lstm/while/Merge_2:1
lstm/while/Merge_3:0
lstm/while/Merge_3:1
lstm/while/NextIteration:0
lstm/while/NextIteration_1:0
lstm/while/NextIteration_2:0
lstm/while/NextIteration_3:0
!lstm/while/ReadVariableOp/Enter:0
lstm/while/ReadVariableOp:0
lstm/while/ReadVariableOp_10:0
lstm/while/ReadVariableOp_11:0
lstm/while/ReadVariableOp_1:0
lstm/while/ReadVariableOp_2:0
lstm/while/ReadVariableOp_3:0
#lstm/while/ReadVariableOp_4/Enter:0
lstm/while/ReadVariableOp_4:0
lstm/while/ReadVariableOp_5:0
lstm/while/ReadVariableOp_6:0
lstm/while/ReadVariableOp_7:0
#lstm/while/ReadVariableOp_8/Enter:0
lstm/while/ReadVariableOp_8:0
lstm/while/ReadVariableOp_9:0
lstm/while/Switch:0
lstm/while/Switch:1
lstm/while/Switch_1:0
lstm/while/Switch_1:1
lstm/while/Switch_2:0
lstm/while/Switch_2:1
lstm/while/Switch_3:0
lstm/while/Switch_3:1
lstm/while/Tanh:0
lstm/while/Tanh_1:0
$lstm/while/TensorArrayReadV3/Enter:0
&lstm/while/TensorArrayReadV3/Enter_1:0
lstm/while/TensorArrayReadV3:0
6lstm/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
0lstm/while/TensorArrayWrite/TensorArrayWriteV3:0
lstm/while/add:0
lstm/while/add_1/y:0
lstm/while/add_1:0
lstm/while/add_2:0
lstm/while/add_3/y:0
lstm/while/add_3:0
lstm/while/add_4:0
lstm/while/add_5:0
lstm/while/add_6:0
lstm/while/add_7/y:0
lstm/while/add_7:0
lstm/while/add_8/y:0
lstm/while/add_8:0
"lstm/while/clip_by_value/Minimum:0
lstm/while/clip_by_value:0
$lstm/while/clip_by_value_1/Minimum:0
lstm/while/clip_by_value_1:0
$lstm/while/clip_by_value_2/Minimum:0
lstm/while/clip_by_value_2:0
lstm/while/mul/x:0
lstm/while/mul:0
lstm/while/mul_1/x:0
lstm/while/mul_1:0
lstm/while/mul_2:0
lstm/while/mul_3:0
lstm/while/mul_4/x:0
lstm/while/mul_4:0
lstm/while/mul_5:0
 lstm/while/strided_slice/stack:0
"lstm/while/strided_slice/stack_1:0
"lstm/while/strided_slice/stack_2:0
lstm/while/strided_slice:0
"lstm/while/strided_slice_1/stack:0
$lstm/while/strided_slice_1/stack_1:0
$lstm/while/strided_slice_1/stack_2:0
#lstm/while/strided_slice_10/stack:0
%lstm/while/strided_slice_10/stack_1:0
%lstm/while/strided_slice_10/stack_2:0
lstm/while/strided_slice_10:0
#lstm/while/strided_slice_11/stack:0
%lstm/while/strided_slice_11/stack_1:0
%lstm/while/strided_slice_11/stack_2:0
lstm/while/strided_slice_11:0
lstm/while/strided_slice_1:0
"lstm/while/strided_slice_2/stack:0
$lstm/while/strided_slice_2/stack_1:0
$lstm/while/strided_slice_2/stack_2:0
lstm/while/strided_slice_2:0
"lstm/while/strided_slice_3/stack:0
$lstm/while/strided_slice_3/stack_1:0
$lstm/while/strided_slice_3/stack_2:0
lstm/while/strided_slice_3:0
"lstm/while/strided_slice_4/stack:0
$lstm/while/strided_slice_4/stack_1:0
$lstm/while/strided_slice_4/stack_2:0
lstm/while/strided_slice_4:0
"lstm/while/strided_slice_5/stack:0
$lstm/while/strided_slice_5/stack_1:0
$lstm/while/strided_slice_5/stack_2:0
lstm/while/strided_slice_5:0
"lstm/while/strided_slice_6/stack:0
$lstm/while/strided_slice_6/stack_1:0
$lstm/while/strided_slice_6/stack_2:0
lstm/while/strided_slice_6:0
"lstm/while/strided_slice_7/stack:0
$lstm/while/strided_slice_7/stack_1:0
$lstm/while/strided_slice_7/stack_2:0
lstm/while/strided_slice_7:0
"lstm/while/strided_slice_8/stack:0
$lstm/while/strided_slice_8/stack_1:0
$lstm/while/strided_slice_8/stack_2:0
lstm/while/strided_slice_8:0
"lstm/while/strided_slice_9/stack:0
$lstm/while/strided_slice_9/stack_1:0
$lstm/while/strided_slice_9/stack_2:0
lstm/while/strided_slice_9:0k
Alstm/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0&lstm/while/TensorArrayReadV3/Enter_1:0<
lstm/TensorArray_1:0$lstm/while/TensorArrayReadV3/Enter:02
lstm/kernel:0!lstm/while/ReadVariableOp/Enter:02
lstm/bias:0#lstm/while/ReadVariableOp_4/Enter:01
lstm/strided_slice_1:0lstm/while/Less/Enter:0L
lstm/TensorArray:06lstm/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0>
lstm/recurrent_kernel:0#lstm/while/ReadVariableOp_8/Enter:0Rlstm/while/Enter:0Rlstm/while/Enter_1:0Rlstm/while/Enter_2:0Rlstm/while/Enter_3:0
Õ+
lstm_1/while/while_context  *lstm_1/while/LoopCond:02lstm_1/while/Merge:0:lstm_1/while/Identity:0Blstm_1/while/Exit:0Blstm_1/while/Exit_1:0Blstm_1/while/Exit_2:0Blstm_1/while/Exit_3:0J°)
lstm_1/TensorArray:0
Clstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
lstm_1/TensorArray_1:0
lstm_1/bias:0
lstm_1/kernel:0
lstm_1/recurrent_kernel:0
lstm_1/strided_slice_1:0
lstm_1/while/BiasAdd:0
lstm_1/while/BiasAdd_1:0
lstm_1/while/BiasAdd_2:0
lstm_1/while/BiasAdd_3:0
lstm_1/while/Const:0
lstm_1/while/Const_1:0
lstm_1/while/Const_2:0
lstm_1/while/Const_3:0
lstm_1/while/Const_4:0
lstm_1/while/Const_5:0
lstm_1/while/Enter:0
lstm_1/while/Enter_1:0
lstm_1/while/Enter_2:0
lstm_1/while/Enter_3:0
lstm_1/while/Exit:0
lstm_1/while/Exit_1:0
lstm_1/while/Exit_2:0
lstm_1/while/Exit_3:0
lstm_1/while/Identity:0
lstm_1/while/Identity_1:0
lstm_1/while/Identity_2:0
lstm_1/while/Identity_3:0
lstm_1/while/Less/Enter:0
lstm_1/while/Less:0
lstm_1/while/LoopCond:0
lstm_1/while/MatMul:0
lstm_1/while/MatMul_1:0
lstm_1/while/MatMul_2:0
lstm_1/while/MatMul_3:0
lstm_1/while/MatMul_4:0
lstm_1/while/MatMul_5:0
lstm_1/while/MatMul_6:0
lstm_1/while/MatMul_7:0
lstm_1/while/Merge:0
lstm_1/while/Merge:1
lstm_1/while/Merge_1:0
lstm_1/while/Merge_1:1
lstm_1/while/Merge_2:0
lstm_1/while/Merge_2:1
lstm_1/while/Merge_3:0
lstm_1/while/Merge_3:1
lstm_1/while/NextIteration:0
lstm_1/while/NextIteration_1:0
lstm_1/while/NextIteration_2:0
lstm_1/while/NextIteration_3:0
#lstm_1/while/ReadVariableOp/Enter:0
lstm_1/while/ReadVariableOp:0
 lstm_1/while/ReadVariableOp_10:0
 lstm_1/while/ReadVariableOp_11:0
lstm_1/while/ReadVariableOp_1:0
lstm_1/while/ReadVariableOp_2:0
lstm_1/while/ReadVariableOp_3:0
%lstm_1/while/ReadVariableOp_4/Enter:0
lstm_1/while/ReadVariableOp_4:0
lstm_1/while/ReadVariableOp_5:0
lstm_1/while/ReadVariableOp_6:0
lstm_1/while/ReadVariableOp_7:0
%lstm_1/while/ReadVariableOp_8/Enter:0
lstm_1/while/ReadVariableOp_8:0
lstm_1/while/ReadVariableOp_9:0
lstm_1/while/Switch:0
lstm_1/while/Switch:1
lstm_1/while/Switch_1:0
lstm_1/while/Switch_1:1
lstm_1/while/Switch_2:0
lstm_1/while/Switch_2:1
lstm_1/while/Switch_3:0
lstm_1/while/Switch_3:1
lstm_1/while/Tanh:0
lstm_1/while/Tanh_1:0
&lstm_1/while/TensorArrayReadV3/Enter:0
(lstm_1/while/TensorArrayReadV3/Enter_1:0
 lstm_1/while/TensorArrayReadV3:0
8lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
2lstm_1/while/TensorArrayWrite/TensorArrayWriteV3:0
lstm_1/while/add:0
lstm_1/while/add_1/y:0
lstm_1/while/add_1:0
lstm_1/while/add_2:0
lstm_1/while/add_3/y:0
lstm_1/while/add_3:0
lstm_1/while/add_4:0
lstm_1/while/add_5:0
lstm_1/while/add_6:0
lstm_1/while/add_7/y:0
lstm_1/while/add_7:0
lstm_1/while/add_8/y:0
lstm_1/while/add_8:0
$lstm_1/while/clip_by_value/Minimum:0
lstm_1/while/clip_by_value:0
&lstm_1/while/clip_by_value_1/Minimum:0
lstm_1/while/clip_by_value_1:0
&lstm_1/while/clip_by_value_2/Minimum:0
lstm_1/while/clip_by_value_2:0
lstm_1/while/mul/x:0
lstm_1/while/mul:0
lstm_1/while/mul_1/x:0
lstm_1/while/mul_1:0
lstm_1/while/mul_2:0
lstm_1/while/mul_3:0
lstm_1/while/mul_4/x:0
lstm_1/while/mul_4:0
lstm_1/while/mul_5:0
"lstm_1/while/strided_slice/stack:0
$lstm_1/while/strided_slice/stack_1:0
$lstm_1/while/strided_slice/stack_2:0
lstm_1/while/strided_slice:0
$lstm_1/while/strided_slice_1/stack:0
&lstm_1/while/strided_slice_1/stack_1:0
&lstm_1/while/strided_slice_1/stack_2:0
%lstm_1/while/strided_slice_10/stack:0
'lstm_1/while/strided_slice_10/stack_1:0
'lstm_1/while/strided_slice_10/stack_2:0
lstm_1/while/strided_slice_10:0
%lstm_1/while/strided_slice_11/stack:0
'lstm_1/while/strided_slice_11/stack_1:0
'lstm_1/while/strided_slice_11/stack_2:0
lstm_1/while/strided_slice_11:0
lstm_1/while/strided_slice_1:0
$lstm_1/while/strided_slice_2/stack:0
&lstm_1/while/strided_slice_2/stack_1:0
&lstm_1/while/strided_slice_2/stack_2:0
lstm_1/while/strided_slice_2:0
$lstm_1/while/strided_slice_3/stack:0
&lstm_1/while/strided_slice_3/stack_1:0
&lstm_1/while/strided_slice_3/stack_2:0
lstm_1/while/strided_slice_3:0
$lstm_1/while/strided_slice_4/stack:0
&lstm_1/while/strided_slice_4/stack_1:0
&lstm_1/while/strided_slice_4/stack_2:0
lstm_1/while/strided_slice_4:0
$lstm_1/while/strided_slice_5/stack:0
&lstm_1/while/strided_slice_5/stack_1:0
&lstm_1/while/strided_slice_5/stack_2:0
lstm_1/while/strided_slice_5:0
$lstm_1/while/strided_slice_6/stack:0
&lstm_1/while/strided_slice_6/stack_1:0
&lstm_1/while/strided_slice_6/stack_2:0
lstm_1/while/strided_slice_6:0
$lstm_1/while/strided_slice_7/stack:0
&lstm_1/while/strided_slice_7/stack_1:0
&lstm_1/while/strided_slice_7/stack_2:0
lstm_1/while/strided_slice_7:0
$lstm_1/while/strided_slice_8/stack:0
&lstm_1/while/strided_slice_8/stack_1:0
&lstm_1/while/strided_slice_8/stack_2:0
lstm_1/while/strided_slice_8:0
$lstm_1/while/strided_slice_9/stack:0
&lstm_1/while/strided_slice_9/stack_1:0
&lstm_1/while/strided_slice_9/stack_2:0
lstm_1/while/strided_slice_9:06
lstm_1/bias:0%lstm_1/while/ReadVariableOp_4/Enter:05
lstm_1/strided_slice_1:0lstm_1/while/Less/Enter:0P
lstm_1/TensorArray:08lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0B
lstm_1/recurrent_kernel:0%lstm_1/while/ReadVariableOp_8/Enter:0@
lstm_1/TensorArray_1:0&lstm_1/while/TensorArrayReadV3/Enter:0o
Clstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0(lstm_1/while/TensorArrayReadV3/Enter_1:06
lstm_1/kernel:0#lstm_1/while/ReadVariableOp/Enter:0Rlstm_1/while/Enter:0Rlstm_1/while/Enter_1:0Rlstm_1/while/Enter_2:0Rlstm_1/while/Enter_3:0
·6
!bidirectional/while/while_context  *bidirectional/while/LoopCond:02bidirectional/while/Merge:0:bidirectional/while/Identity:0Bbidirectional/while/Exit:0Bbidirectional/while/Exit_1:0Bbidirectional/while/Exit_2:0Bbidirectional/while/Exit_3:0J¾3
bidirectional/TensorArray:0
Jbidirectional/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
bidirectional/TensorArray_1:0
#bidirectional/forward_lstm_2/bias:0
%bidirectional/forward_lstm_2/kernel:0
/bidirectional/forward_lstm_2/recurrent_kernel:0
bidirectional/strided_slice_1:0
bidirectional/while/BiasAdd:0
bidirectional/while/BiasAdd_1:0
bidirectional/while/BiasAdd_2:0
bidirectional/while/BiasAdd_3:0
bidirectional/while/Const:0
bidirectional/while/Const_1:0
bidirectional/while/Const_2:0
bidirectional/while/Const_3:0
bidirectional/while/Const_4:0
bidirectional/while/Const_5:0
bidirectional/while/Enter:0
bidirectional/while/Enter_1:0
bidirectional/while/Enter_2:0
bidirectional/while/Enter_3:0
bidirectional/while/Exit:0
bidirectional/while/Exit_1:0
bidirectional/while/Exit_2:0
bidirectional/while/Exit_3:0
bidirectional/while/Identity:0
 bidirectional/while/Identity_1:0
 bidirectional/while/Identity_2:0
 bidirectional/while/Identity_3:0
 bidirectional/while/Less/Enter:0
bidirectional/while/Less:0
bidirectional/while/LoopCond:0
bidirectional/while/MatMul:0
bidirectional/while/MatMul_1:0
bidirectional/while/MatMul_2:0
bidirectional/while/MatMul_3:0
bidirectional/while/MatMul_4:0
bidirectional/while/MatMul_5:0
bidirectional/while/MatMul_6:0
bidirectional/while/MatMul_7:0
bidirectional/while/Merge:0
bidirectional/while/Merge:1
bidirectional/while/Merge_1:0
bidirectional/while/Merge_1:1
bidirectional/while/Merge_2:0
bidirectional/while/Merge_2:1
bidirectional/while/Merge_3:0
bidirectional/while/Merge_3:1
#bidirectional/while/NextIteration:0
%bidirectional/while/NextIteration_1:0
%bidirectional/while/NextIteration_2:0
%bidirectional/while/NextIteration_3:0
*bidirectional/while/ReadVariableOp/Enter:0
$bidirectional/while/ReadVariableOp:0
'bidirectional/while/ReadVariableOp_10:0
'bidirectional/while/ReadVariableOp_11:0
&bidirectional/while/ReadVariableOp_1:0
&bidirectional/while/ReadVariableOp_2:0
&bidirectional/while/ReadVariableOp_3:0
,bidirectional/while/ReadVariableOp_4/Enter:0
&bidirectional/while/ReadVariableOp_4:0
&bidirectional/while/ReadVariableOp_5:0
&bidirectional/while/ReadVariableOp_6:0
&bidirectional/while/ReadVariableOp_7:0
,bidirectional/while/ReadVariableOp_8/Enter:0
&bidirectional/while/ReadVariableOp_8:0
&bidirectional/while/ReadVariableOp_9:0
bidirectional/while/Switch:0
bidirectional/while/Switch:1
bidirectional/while/Switch_1:0
bidirectional/while/Switch_1:1
bidirectional/while/Switch_2:0
bidirectional/while/Switch_2:1
bidirectional/while/Switch_3:0
bidirectional/while/Switch_3:1
bidirectional/while/Tanh:0
bidirectional/while/Tanh_1:0
-bidirectional/while/TensorArrayReadV3/Enter:0
/bidirectional/while/TensorArrayReadV3/Enter_1:0
'bidirectional/while/TensorArrayReadV3:0
?bidirectional/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
9bidirectional/while/TensorArrayWrite/TensorArrayWriteV3:0
bidirectional/while/add:0
bidirectional/while/add_1/y:0
bidirectional/while/add_1:0
bidirectional/while/add_2:0
bidirectional/while/add_3/y:0
bidirectional/while/add_3:0
bidirectional/while/add_4:0
bidirectional/while/add_5:0
bidirectional/while/add_6:0
bidirectional/while/add_7/y:0
bidirectional/while/add_7:0
bidirectional/while/add_8/y:0
bidirectional/while/add_8:0
+bidirectional/while/clip_by_value/Minimum:0
#bidirectional/while/clip_by_value:0
-bidirectional/while/clip_by_value_1/Minimum:0
%bidirectional/while/clip_by_value_1:0
-bidirectional/while/clip_by_value_2/Minimum:0
%bidirectional/while/clip_by_value_2:0
bidirectional/while/mul/x:0
bidirectional/while/mul:0
bidirectional/while/mul_1/x:0
bidirectional/while/mul_1:0
bidirectional/while/mul_2:0
bidirectional/while/mul_3:0
bidirectional/while/mul_4/x:0
bidirectional/while/mul_4:0
bidirectional/while/mul_5:0
)bidirectional/while/strided_slice/stack:0
+bidirectional/while/strided_slice/stack_1:0
+bidirectional/while/strided_slice/stack_2:0
#bidirectional/while/strided_slice:0
+bidirectional/while/strided_slice_1/stack:0
-bidirectional/while/strided_slice_1/stack_1:0
-bidirectional/while/strided_slice_1/stack_2:0
,bidirectional/while/strided_slice_10/stack:0
.bidirectional/while/strided_slice_10/stack_1:0
.bidirectional/while/strided_slice_10/stack_2:0
&bidirectional/while/strided_slice_10:0
,bidirectional/while/strided_slice_11/stack:0
.bidirectional/while/strided_slice_11/stack_1:0
.bidirectional/while/strided_slice_11/stack_2:0
&bidirectional/while/strided_slice_11:0
%bidirectional/while/strided_slice_1:0
+bidirectional/while/strided_slice_2/stack:0
-bidirectional/while/strided_slice_2/stack_1:0
-bidirectional/while/strided_slice_2/stack_2:0
%bidirectional/while/strided_slice_2:0
+bidirectional/while/strided_slice_3/stack:0
-bidirectional/while/strided_slice_3/stack_1:0
-bidirectional/while/strided_slice_3/stack_2:0
%bidirectional/while/strided_slice_3:0
+bidirectional/while/strided_slice_4/stack:0
-bidirectional/while/strided_slice_4/stack_1:0
-bidirectional/while/strided_slice_4/stack_2:0
%bidirectional/while/strided_slice_4:0
+bidirectional/while/strided_slice_5/stack:0
-bidirectional/while/strided_slice_5/stack_1:0
-bidirectional/while/strided_slice_5/stack_2:0
%bidirectional/while/strided_slice_5:0
+bidirectional/while/strided_slice_6/stack:0
-bidirectional/while/strided_slice_6/stack_1:0
-bidirectional/while/strided_slice_6/stack_2:0
%bidirectional/while/strided_slice_6:0
+bidirectional/while/strided_slice_7/stack:0
-bidirectional/while/strided_slice_7/stack_1:0
-bidirectional/while/strided_slice_7/stack_2:0
%bidirectional/while/strided_slice_7:0
+bidirectional/while/strided_slice_8/stack:0
-bidirectional/while/strided_slice_8/stack_1:0
-bidirectional/while/strided_slice_8/stack_2:0
%bidirectional/while/strided_slice_8:0
+bidirectional/while/strided_slice_9/stack:0
-bidirectional/while/strided_slice_9/stack_1:0
-bidirectional/while/strided_slice_9/stack_2:0
%bidirectional/while/strided_slice_9:0C
bidirectional/strided_slice_1:0 bidirectional/while/Less/Enter:0N
bidirectional/TensorArray_1:0-bidirectional/while/TensorArrayReadV3/Enter:0S
#bidirectional/forward_lstm_2/bias:0,bidirectional/while/ReadVariableOp_4/Enter:0}
Jbidirectional/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0/bidirectional/while/TensorArrayReadV3/Enter_1:0^
bidirectional/TensorArray:0?bidirectional/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0S
%bidirectional/forward_lstm_2/kernel:0*bidirectional/while/ReadVariableOp/Enter:0_
/bidirectional/forward_lstm_2/recurrent_kernel:0,bidirectional/while/ReadVariableOp_8/Enter:0Rbidirectional/while/Enter:0Rbidirectional/while/Enter_1:0Rbidirectional/while/Enter_2:0Rbidirectional/while/Enter_3:0
9
#bidirectional/while_1/while_context  * bidirectional/while_1/LoopCond:02bidirectional/while_1/Merge:0: bidirectional/while_1/Identity:0Bbidirectional/while_1/Exit:0Bbidirectional/while_1/Exit_1:0Bbidirectional/while_1/Exit_2:0Bbidirectional/while_1/Exit_3:0J6
Lbidirectional/TensorArrayUnstack_1/TensorArrayScatter/TensorArrayScatterV3:0
bidirectional/TensorArray_2:0
bidirectional/TensorArray_3:0
$bidirectional/backward_lstm_2/bias:0
&bidirectional/backward_lstm_2/kernel:0
0bidirectional/backward_lstm_2/recurrent_kernel:0
 bidirectional/strided_slice_16:0
bidirectional/while_1/BiasAdd:0
!bidirectional/while_1/BiasAdd_1:0
!bidirectional/while_1/BiasAdd_2:0
!bidirectional/while_1/BiasAdd_3:0
bidirectional/while_1/Const:0
bidirectional/while_1/Const_1:0
bidirectional/while_1/Const_2:0
bidirectional/while_1/Const_3:0
bidirectional/while_1/Const_4:0
bidirectional/while_1/Const_5:0
bidirectional/while_1/Enter:0
bidirectional/while_1/Enter_1:0
bidirectional/while_1/Enter_2:0
bidirectional/while_1/Enter_3:0
bidirectional/while_1/Exit:0
bidirectional/while_1/Exit_1:0
bidirectional/while_1/Exit_2:0
bidirectional/while_1/Exit_3:0
 bidirectional/while_1/Identity:0
"bidirectional/while_1/Identity_1:0
"bidirectional/while_1/Identity_2:0
"bidirectional/while_1/Identity_3:0
"bidirectional/while_1/Less/Enter:0
bidirectional/while_1/Less:0
 bidirectional/while_1/LoopCond:0
bidirectional/while_1/MatMul:0
 bidirectional/while_1/MatMul_1:0
 bidirectional/while_1/MatMul_2:0
 bidirectional/while_1/MatMul_3:0
 bidirectional/while_1/MatMul_4:0
 bidirectional/while_1/MatMul_5:0
 bidirectional/while_1/MatMul_6:0
 bidirectional/while_1/MatMul_7:0
bidirectional/while_1/Merge:0
bidirectional/while_1/Merge:1
bidirectional/while_1/Merge_1:0
bidirectional/while_1/Merge_1:1
bidirectional/while_1/Merge_2:0
bidirectional/while_1/Merge_2:1
bidirectional/while_1/Merge_3:0
bidirectional/while_1/Merge_3:1
%bidirectional/while_1/NextIteration:0
'bidirectional/while_1/NextIteration_1:0
'bidirectional/while_1/NextIteration_2:0
'bidirectional/while_1/NextIteration_3:0
,bidirectional/while_1/ReadVariableOp/Enter:0
&bidirectional/while_1/ReadVariableOp:0
)bidirectional/while_1/ReadVariableOp_10:0
)bidirectional/while_1/ReadVariableOp_11:0
(bidirectional/while_1/ReadVariableOp_1:0
(bidirectional/while_1/ReadVariableOp_2:0
(bidirectional/while_1/ReadVariableOp_3:0
.bidirectional/while_1/ReadVariableOp_4/Enter:0
(bidirectional/while_1/ReadVariableOp_4:0
(bidirectional/while_1/ReadVariableOp_5:0
(bidirectional/while_1/ReadVariableOp_6:0
(bidirectional/while_1/ReadVariableOp_7:0
.bidirectional/while_1/ReadVariableOp_8/Enter:0
(bidirectional/while_1/ReadVariableOp_8:0
(bidirectional/while_1/ReadVariableOp_9:0
bidirectional/while_1/Switch:0
bidirectional/while_1/Switch:1
 bidirectional/while_1/Switch_1:0
 bidirectional/while_1/Switch_1:1
 bidirectional/while_1/Switch_2:0
 bidirectional/while_1/Switch_2:1
 bidirectional/while_1/Switch_3:0
 bidirectional/while_1/Switch_3:1
bidirectional/while_1/Tanh:0
bidirectional/while_1/Tanh_1:0
/bidirectional/while_1/TensorArrayReadV3/Enter:0
1bidirectional/while_1/TensorArrayReadV3/Enter_1:0
)bidirectional/while_1/TensorArrayReadV3:0
Abidirectional/while_1/TensorArrayWrite/TensorArrayWriteV3/Enter:0
;bidirectional/while_1/TensorArrayWrite/TensorArrayWriteV3:0
bidirectional/while_1/add:0
bidirectional/while_1/add_1/y:0
bidirectional/while_1/add_1:0
bidirectional/while_1/add_2:0
bidirectional/while_1/add_3/y:0
bidirectional/while_1/add_3:0
bidirectional/while_1/add_4:0
bidirectional/while_1/add_5:0
bidirectional/while_1/add_6:0
bidirectional/while_1/add_7/y:0
bidirectional/while_1/add_7:0
bidirectional/while_1/add_8/y:0
bidirectional/while_1/add_8:0
-bidirectional/while_1/clip_by_value/Minimum:0
%bidirectional/while_1/clip_by_value:0
/bidirectional/while_1/clip_by_value_1/Minimum:0
'bidirectional/while_1/clip_by_value_1:0
/bidirectional/while_1/clip_by_value_2/Minimum:0
'bidirectional/while_1/clip_by_value_2:0
bidirectional/while_1/mul/x:0
bidirectional/while_1/mul:0
bidirectional/while_1/mul_1/x:0
bidirectional/while_1/mul_1:0
bidirectional/while_1/mul_2:0
bidirectional/while_1/mul_3:0
bidirectional/while_1/mul_4/x:0
bidirectional/while_1/mul_4:0
bidirectional/while_1/mul_5:0
+bidirectional/while_1/strided_slice/stack:0
-bidirectional/while_1/strided_slice/stack_1:0
-bidirectional/while_1/strided_slice/stack_2:0
%bidirectional/while_1/strided_slice:0
-bidirectional/while_1/strided_slice_1/stack:0
/bidirectional/while_1/strided_slice_1/stack_1:0
/bidirectional/while_1/strided_slice_1/stack_2:0
.bidirectional/while_1/strided_slice_10/stack:0
0bidirectional/while_1/strided_slice_10/stack_1:0
0bidirectional/while_1/strided_slice_10/stack_2:0
(bidirectional/while_1/strided_slice_10:0
.bidirectional/while_1/strided_slice_11/stack:0
0bidirectional/while_1/strided_slice_11/stack_1:0
0bidirectional/while_1/strided_slice_11/stack_2:0
(bidirectional/while_1/strided_slice_11:0
'bidirectional/while_1/strided_slice_1:0
-bidirectional/while_1/strided_slice_2/stack:0
/bidirectional/while_1/strided_slice_2/stack_1:0
/bidirectional/while_1/strided_slice_2/stack_2:0
'bidirectional/while_1/strided_slice_2:0
-bidirectional/while_1/strided_slice_3/stack:0
/bidirectional/while_1/strided_slice_3/stack_1:0
/bidirectional/while_1/strided_slice_3/stack_2:0
'bidirectional/while_1/strided_slice_3:0
-bidirectional/while_1/strided_slice_4/stack:0
/bidirectional/while_1/strided_slice_4/stack_1:0
/bidirectional/while_1/strided_slice_4/stack_2:0
'bidirectional/while_1/strided_slice_4:0
-bidirectional/while_1/strided_slice_5/stack:0
/bidirectional/while_1/strided_slice_5/stack_1:0
/bidirectional/while_1/strided_slice_5/stack_2:0
'bidirectional/while_1/strided_slice_5:0
-bidirectional/while_1/strided_slice_6/stack:0
/bidirectional/while_1/strided_slice_6/stack_1:0
/bidirectional/while_1/strided_slice_6/stack_2:0
'bidirectional/while_1/strided_slice_6:0
-bidirectional/while_1/strided_slice_7/stack:0
/bidirectional/while_1/strided_slice_7/stack_1:0
/bidirectional/while_1/strided_slice_7/stack_2:0
'bidirectional/while_1/strided_slice_7:0
-bidirectional/while_1/strided_slice_8/stack:0
/bidirectional/while_1/strided_slice_8/stack_1:0
/bidirectional/while_1/strided_slice_8/stack_2:0
'bidirectional/while_1/strided_slice_8:0
-bidirectional/while_1/strided_slice_9/stack:0
/bidirectional/while_1/strided_slice_9/stack_1:0
/bidirectional/while_1/strided_slice_9/stack_2:0
'bidirectional/while_1/strided_slice_9:0
Lbidirectional/TensorArrayUnstack_1/TensorArrayScatter/TensorArrayScatterV3:01bidirectional/while_1/TensorArrayReadV3/Enter_1:0b
bidirectional/TensorArray_2:0Abidirectional/while_1/TensorArrayWrite/TensorArrayWriteV3/Enter:0V
$bidirectional/backward_lstm_2/bias:0.bidirectional/while_1/ReadVariableOp_4/Enter:0F
 bidirectional/strided_slice_16:0"bidirectional/while_1/Less/Enter:0V
&bidirectional/backward_lstm_2/kernel:0,bidirectional/while_1/ReadVariableOp/Enter:0P
bidirectional/TensorArray_3:0/bidirectional/while_1/TensorArrayReadV3/Enter:0b
0bidirectional/backward_lstm_2/recurrent_kernel:0.bidirectional/while_1/ReadVariableOp_8/Enter:0Rbidirectional/while_1/Enter:0Rbidirectional/while_1/Enter_1:0Rbidirectional/while_1/Enter_2:0Rbidirectional/while_1/Enter_3:0"Ń
trainable_variables¹¶
 
embedding_1/embeddings:0embedding_1/embeddings/Assign,embedding_1/embeddings/Read/ReadVariableOp:0(23embedding_1/embeddings/Initializer/random_uniform:08
t
lstm/kernel:0lstm/kernel/Assign!lstm/kernel/Read/ReadVariableOp:0(2(lstm/kernel/Initializer/random_uniform:08

lstm/recurrent_kernel:0lstm/recurrent_kernel/Assign+lstm/recurrent_kernel/Read/ReadVariableOp:0(2)lstm/recurrent_kernel/Initializer/mul_1:08
d
lstm/bias:0lstm/bias/Assignlstm/bias/Read/ReadVariableOp:0(2lstm/bias/Initializer/concat:08
|
lstm_1/kernel:0lstm_1/kernel/Assign#lstm_1/kernel/Read/ReadVariableOp:0(2*lstm_1/kernel/Initializer/random_uniform:08

lstm_1/recurrent_kernel:0lstm_1/recurrent_kernel/Assign-lstm_1/recurrent_kernel/Read/ReadVariableOp:0(2+lstm_1/recurrent_kernel/Initializer/mul_1:08
l
lstm_1/bias:0lstm_1/bias/Assign!lstm_1/bias/Read/ReadVariableOp:0(2 lstm_1/bias/Initializer/concat:08
Ō
%bidirectional/forward_lstm_2/kernel:0*bidirectional/forward_lstm_2/kernel/Assign9bidirectional/forward_lstm_2/kernel/Read/ReadVariableOp:0(2@bidirectional/forward_lstm_2/kernel/Initializer/random_uniform:08
ó
/bidirectional/forward_lstm_2/recurrent_kernel:04bidirectional/forward_lstm_2/recurrent_kernel/AssignCbidirectional/forward_lstm_2/recurrent_kernel/Read/ReadVariableOp:0(2Abidirectional/forward_lstm_2/recurrent_kernel/Initializer/mul_1:08
Ä
#bidirectional/forward_lstm_2/bias:0(bidirectional/forward_lstm_2/bias/Assign7bidirectional/forward_lstm_2/bias/Read/ReadVariableOp:0(26bidirectional/forward_lstm_2/bias/Initializer/concat:08
Ų
&bidirectional/backward_lstm_2/kernel:0+bidirectional/backward_lstm_2/kernel/Assign:bidirectional/backward_lstm_2/kernel/Read/ReadVariableOp:0(2Abidirectional/backward_lstm_2/kernel/Initializer/random_uniform:08
÷
0bidirectional/backward_lstm_2/recurrent_kernel:05bidirectional/backward_lstm_2/recurrent_kernel/AssignDbidirectional/backward_lstm_2/recurrent_kernel/Read/ReadVariableOp:0(2Bbidirectional/backward_lstm_2/recurrent_kernel/Initializer/mul_1:08
Č
$bidirectional/backward_lstm_2/bias:0)bidirectional/backward_lstm_2/bias/Assign8bidirectional/backward_lstm_2/bias/Read/ReadVariableOp:0(27bidirectional/backward_lstm_2/bias/Initializer/concat:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08*
serving_default’
<
word_input:0,
word_input:0’’’’’’’’’’’’’’’’’’
I
char_input:09
char_input:0'’’’’’’’’’’’’’’’’’’’’’’’’’’’X
predict_output/truediv:0<
predict_output/truediv:0’’’’’’’’’’’’’’’’’’tensorflow/serving/predict