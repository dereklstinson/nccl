package nccl

/*
#include "nccl.h"
#include <cuda_runtime.h>
*/
import "C"
import (
	"github.com/dereklstinson/cutil"
	"unsafe"
)

//Streamer should be a stream from cuda or cudart.
//
//Sync is not used here, but since sync is a part of every cuda stream I put this here
//
//Ptr() returns an unsafe.Pointer of the stream.
type Streamer interface {
	Sync() error
	Ptr() unsafe.Pointer
}

//Reduce - Reduces data arrays of length count in sendbuff into recvbuff using op operation.
// recvbuff may be NULL on all calls except for root device.
// root is the rank (not the CUDA device) where data will reside after the
// operation is complete.
//
// In-place operation will happen if sendbuff == recvbuff.
func Reduce(sendbuff, recvbuff cutil.Mem,
	count uint,
	d DataType,
	op RedOp,
	root int32,
	c *Comm,
	s Streamer) (err error) {
	if recvbuff == nil {
		err = result(C.ncclReduce(
			sendbuff.Ptr(),
			nil,
			(C.size_t)(count),
			d.c(),
			op.c(),
			(C.int)(root),
			c.com,
			(C.cudaStream_t)(s.Ptr()))).error("Reduce")
	}
	err = result(C.ncclReduce(
		sendbuff.Ptr(),
		recvbuff.Ptr(),
		(C.size_t)(count),
		d.c(),
		op.c(),
		(C.int)(root),
		c.com,
		(C.cudaStream_t)(s.Ptr()))).error("Reduce")

	return err
}

// Broadcast - Copies count values from root to all other devices.
// root is the rank (not the CUDA device) where data resides before the
// operation is started.
//
// In-place operation will happen if sendbuff == recvbuff.
//
func Broadcast(sendbuff, recvbuff cutil.Mem,
	count uint,
	d DataType,
	root int32,
	c *Comm,
	s Streamer) (err error) {
	err = result(C.ncclBroadcast(
		sendbuff.Ptr(),
		recvbuff.Ptr(),
		(C.size_t)(count),
		d.c(),
		(C.int)(root),
		c.com,
		(C.cudaStream_t)(s.Ptr()))).error("Broadcast")

	return err
}

//AllReduce reduces data arrays of length count in sendbuff using op operation, and
//leaves identical copies of result on each recvbuff.
//
//In-place operation will happen if sendbuff == recvbuff.
func AllReduce(sendbuff, recvbuff cutil.Mem,
	count uint,
	d DataType,
	op RedOp,
	c *Comm,
	s Streamer) (err error) {
	err = result(C.ncclAllReduce(
		sendbuff.Ptr(),
		recvbuff.Ptr(),
		(C.size_t)(count),
		d.c(),
		op.c(),
		c.com,
		(C.cudaStream_t)(s.Ptr()))).error("AllReduce")
	return err
}

//ReduceScatter reduces data in sendbuff using op operation and leaves reduced result
//scattered over the devices so that recvbuff on rank i will contain the i-th
//block of the result.
//Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
//should have a size of at least nranks*recvcount elements.
//
//In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
func ReduceScatter(sendbuff, recvbuff cutil.Mem,
	recvcount uint,
	d DataType,
	op RedOp,
	c *Comm,
	s Streamer) (err error) {

	err = result(C.ncclReduceScatter(
		sendbuff.Ptr(),
		recvbuff.Ptr(),
		(C.size_t)(recvcount),
		d.c(),
		op.c(),
		c.com,
		(C.cudaStream_t)(s.Ptr()))).error("ReduceScatter")
	return err
}

//AllGather - Each device gathers sendcount values from other GPUs into recvbuff,
//receiving data from rank i at offset i*sendcount.
//Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
//should have a size of at least nranks*sendcount elements.
//
//In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
func AllGather(sendbuff, recvbuff cutil.Mem,
	sendcount uint,
	d DataType,
	c *Comm,
	s Streamer) (err error) {

	err = result(C.ncclAllGather(
		sendbuff.Ptr(),
		recvbuff.Ptr(),
		(C.size_t)(sendcount),
		d.c(),
		c.com,
		(C.cudaStream_t)(s.Ptr()))).error("AllGather")
	return err
}

//GroupStart - Start a group call. All subsequent calls to NCCL may not block due to
//inter-CPU synchronization.
//
//When managing multiple GPUs from a single thread, and since NCCL collective
//calls may perform inter-CPU synchronization, we need to "group" calls for
//different ranks/devices into a single call.
//
//Grouping NCCL calls as being part of the same collective operation is done
//using ncclGroupStart and ncclGroupEnd. ncclGroupStart will enqueue all
//collective calls until the ncclGroupEnd call, which will wait for all calls
//to be complete. Note that for collective communication, ncclGroupEnd only
//guarantees that the operations are enqueued on the streams, not that
//the operation is effectively done.
//
//Both collective communication and ncclCommInitRank can be used in conjunction
//of ncclGroupStart/ncclGroupEnd.
func GroupStart() (err error) {
	err = result(C.ncclGroupStart()).error("GroupStart")
	return err
}

//GroupEnd - End a group call. Wait for all calls since ncclGroupStart to complete
//before returning.
func GroupEnd() (err error) {
	err = result(C.ncclGroupEnd()).error("GroupStart")
	return err
}
