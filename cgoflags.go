package nccl

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo CFLAGS: -I/usr/include/
#cgo LDFLAGS: -L/usr/lib/x86_64-linux-gnu -lnccl
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -lrt
*/
import "C"
