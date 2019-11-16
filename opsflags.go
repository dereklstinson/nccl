package nccl

/*
#include "nccl.h"
*/
import "C"

//RedOp - Reduction operation selector
type RedOp C.ncclRedOp_t

func (r RedOp) c() C.ncclRedOp_t {
	return (C.ncclRedOp_t)(r)
}

//Sum is a flag method changes inner value and returns that value.
func (r *RedOp) Sum() RedOp {
	*r = (RedOp)(C.ncclSum)
	return *r
}

//Prod is a flag method changes inner value and returns that value.
func (r *RedOp) Prod() RedOp {
	*r = (RedOp)(C.ncclProd)
	return *r
}

//Max is a flag method changes inner value and returns that value.
func (r *RedOp) Max() RedOp {
	*r = (RedOp)(C.ncclMax)
	return *r
}

//Min is a flag method changes inner value and returns that value.
func (r *RedOp) Min() RedOp {
	*r = (RedOp)(C.ncclMin)
	return *r
}

//NumOps is a flag method changes inner value and returns that value.
func (r *RedOp) NumOps() RedOp {
	*r = (RedOp)(C.ncclNumOps)
	return *r
}

//DataType Data types some of them are the same
type DataType C.ncclDataType_t

func (d DataType) c() C.ncclDataType_t {
	return (C.ncclDataType_t)(d)
}

//Int8 is a flag for DataType method will change value of d and return that value.
//
//Same as Char
func (d *DataType) Int8() DataType {
	*d = (DataType)(C.ncclInt8)
	return *d
}

//Char is a flag for DataType method will change value of d and return that value.
//
//Same as Int8
func (d *DataType) Char() DataType {
	*d = (DataType)(C.ncclChar)
	return *d
}

//Uint8 is a flag for DataType method will change value of d and return that value.
func (d *DataType) Uint8() DataType {
	*d = (DataType)(C.ncclUint8)
	return *d
}

//Int32 is a flag for DataType method will change value of d and return that value.
//
//Same as Int
func (d *DataType) Int32() DataType {
	*d = (DataType)(C.ncclInt32)
	return *d
}

//Int is a flag for DataType method will change value of d and return that value.
//
//Same as Int32
func (d *DataType) Int() DataType {
	*d = (DataType)(C.ncclInt)
	return *d
}

//Uint32 is a flag for DataType method will change value of d and return that value.
func (d *DataType) Uint32() DataType {
	*d = (DataType)(C.ncclUint32)
	return *d
}

//Int64 is a flag for DataType method will change value of d and return that value.
func (d *DataType) Int64() DataType {
	*d = (DataType)(C.ncclInt64)
	return *d
}

//Uint64 is a flag for DataType method will change value of d and return that value.
func (d *DataType) Uint64() DataType {
	*d = (DataType)(C.ncclUint64)
	return *d
}

//Float16 is a flag for DataType method will change value of d and return that value.
//
//Same as Half
func (d *DataType) Float16() DataType {
	*d = (DataType)(C.ncclFloat16)
	return *d
}

//Half is a flag for DataType method will change value of d and return that value.
//
//Same as Float16
func (d *DataType) Half() DataType {
	*d = (DataType)(C.ncclHalf)
	return *d
}

//Float32 is a flag for DataType method will change value of d and return that value.
//
//Same as Float
func (d *DataType) Float32() DataType {
	*d = (DataType)(C.ncclFloat32)
	return *d
}

//Float is a flag for DataType method will change value of d and return that value.
//
//Same as Float32
func (d *DataType) Float() DataType {
	*d = (DataType)(C.ncclFloat)
	return *d
}

//Float64 is a flag for DataType method will change value of d and return that value.
//
//Same as Double
func (d *DataType) Float64() DataType {
	*d = (DataType)(C.ncclFloat64)
	return *d
}

//Double is a flag for DataType method will change value of d and return that value.
//
//Same as Float64
func (d *DataType) Double() DataType {
	*d = (DataType)(C.ncclDouble)
	return *d
}

//NumTypes is a flag for DataType method will change value of d and return that value.
func (d *DataType) NumTypes() DataType {
	*d = (DataType)(C.ncclNumTypes)
	return *d
}
