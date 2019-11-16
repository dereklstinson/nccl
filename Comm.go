package nccl

/*
#include "nccl.h"
*/
import "C"
import (
	"runtime"
)

//Comm - Opaque handle to communicator
type Comm struct {
	aborted bool
	com     C.ncclComm_t
}

/*
func commtoC() []C.ncclComm_t{

}
*/
func ctoComm(cs []C.ncclComm_t) (comms []*Comm) {
	comms = make([]*Comm, len(cs))
	for i := range cs {
		comms[i].com = cs[i]
	}
	return comms
}

//UniqueID for for nccl
type UniqueID C.ncclUniqueId

func (u UniqueID) c() C.ncclUniqueId { return (C.ncclUniqueId)(u) }

//GetUniqueID - from nccl.h
//
//Generates an Id to be used in ncclCommInitRank. ncclGetUniqueId should be
//called once and the Id should be distributed to all ranks in the
//communicator before calling ncclCommInitRank.
func GetUniqueID() (u UniqueID, err error) {
	err = result(C.ncclGetUniqueId((*C.ncclUniqueId)(&u))).error("GetUniqueID")
	return u, err
}

// CommInitRank - nccl.h comment below
//
// Creates a new communicator (multi thread/process version).
// rank must be between 0 and nranks-1 and unique within a communicator clique.
// Each rank is associated to a CUDA device, which has to be set before calling
// ncclCommInitRank.
// ncclCommInitRank implicitly syncronizes with other ranks, so it must be
// called by different threads/processes or use ncclGroupStart/ncclGroupEnd.
func CommInitRank(nrank int32, commID UniqueID, rank int32) (c *Comm, err error) {
	c = new(Comm)
	err = result(C.ncclCommInitRank(&c.com, (C.int)(nrank), commID.c(), (C.int)(rank))).error("CommInitRank")
	runtime.SetFinalizer(c, ncclCommDestroy)
	return c, nil
}

// ComInitAll - Creates a clique of communicators (single process version).
// This is a convenience function to create a single-process communicator clique.
// Returns an array of ndev newly initialized communicators in comm.
func ComInitAll(devlist []int32) (comms []*Comm, err error) {
	cs := make([]C.ncclComm_t, len(devlist))
	devc := int32tocarray(devlist)
	ndev := C.int(len(devlist))
	err = result(C.ncclCommInitAll(&cs[0], ndev, &devc[0])).error("ComInitAll")
	comms = ctoComm(cs)
	for i := range comms {
		runtime.SetFinalizer(comms[i], ncclCommDestroy)
	}
	return
}
func ncclCommDestroy(c *Comm) error {
	if c.aborted {
		return nil
	}
	return result(C.ncclCommDestroy(c.com)).error("ncclCommDestroy")
}
func int32tocarray(a []int32) []C.int {
	b := make([]C.int, len(a))
	for i := range a {
		b[i] = (C.int)(a[i])
	}
	return b
}

//Abort - Frees resources associated with communicator object and aborts any operations
//that might still be running on the device.  This will bypass the GC, but it shouldn't run to any problems with it.
//It raises a flag telling the GC it is aborted.
func (c *Comm) Abort() error {
	c.aborted = true
	return result(C.ncclCommAbort(c.com)).error("(c *Comm)Abort()")
}

//GetAsyncError - Checks whether the comm has encountered any asynchronous errors */
func (c *Comm) GetAsyncError() (fromfunc, fromasync error) {
	var async result
	fromfunc = result(C.ncclCommGetAsyncError(c.com, async.cptr())).error("GetAsyncError")
	return fromfunc, async.error("GetAsyncError")
}

//Count - Gets the number of ranks in the communicator clique.
func (c *Comm) Count() (n int32, err error) {
	err = result(C.ncclCommCount(c.com, (*C.int)(&n))).error("Count")
	return n, err

}

//CuDevice - returns the cuda device number associated with the communicator.
func (c *Comm) CuDevice() (dev int32, err error) {
	err = result(C.ncclCommCuDevice(c.com, (*C.int)(&dev))).error("CuDevice")
	return dev, err
}

//Rank - returns the user-ordered "rank" associated with the communicator.
func (c *Comm) Rank() (r int32, err error) {
	err = result(C.ncclCommUserRank(c.com, (*C.int)(&r))).error("Rank")
	return r, err
}
