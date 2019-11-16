package nccl

/*
#include "nccl.h"
*/
import "C"
import (
	"errors"
)

//GoPackageNCCLVersion is nccl version this package was built on.  If using other version, there might be conflicts.
func GoPackageNCCLVersion() int {
	return 2408
}

// GetVersion - From nccl header.
//
// Return the NCCL_VERSION_CODE of the NCCL library in the supplied integer.
// This integer is coded with the MAJOR, MINOR and PATCH level of the
// NCCL library
func GetVersion() (version int32, err error) {
	err = result(C.ncclGetVersion((*C.int)(&version))).error("GetVersion")
	return version, err
}

type result C.ncclResult_t

func (r *result) cptr() *C.ncclResult_t {
	return (*C.ncclResult_t)(r)
}
func (r result) c() C.ncclResult_t {
	return (C.ncclResult_t(r))
}
func (r result) error(comment string) error {
	if (C.ncclResult_t)(r) == C.ncclSuccess {
		return nil
	}
	return errors.New(comment + " : " + r.Error())
}

func (r result) Error() string {

	switch (C.ncclResult_t)(r) {
	case C.ncclSuccess:
		return "ncclSuccess"
	case C.ncclUnhandledCudaError:
		return "ncclUnhandledCudaError"
	case C.ncclSystemError:
		return "ncclSystemError"
	case C.ncclInternalError:
		return "ncclInternalError"
	case C.ncclInvalidArgument:
		return "ncclInvalidArgument"
	case C.ncclInvalidUsage:
		return "ncclInvalidUsage"
	case C.ncclNumResults:
		return "ncclNumResults"
	}
	return "Undefined NCCL Error in go"
}
