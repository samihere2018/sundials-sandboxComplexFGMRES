/* -----------------------------------------------------------------
 * Programmer(s): Mustafa Aggul @ SMU
 * -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2024, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------
 * This is the implementation file for a complex N_Vector
 * (note that this template just implements the serial N_Vector).
 * -----------------------------------------------------------------*/

#ifndef _CS_NVECTOR_H
#define _CS_NVECTOR_H

#define SUNCabs(x) (cabs((x))) //define for other precisions too!
#define SUNCsqrt(x) ((creal(csqrt((x)))) <= SUN_RCONST(0.0) ? (-csqrt((x))) : (csqrt((x)))) //returns the sqrt(x) with positive real part

#include <stdio.h>
#include <complex.h>
#include <sundials/sundials_nvector.h>

/* The following type definition should be moved to sundials_types.h header file*/
#define suncomplextype double complex

#ifdef __cplusplus /* wrapper to enable C++ usage */
extern "C" {
#endif

/*
 * -----------------------------------------------------------------
 * Custom implementation of N_Vector
 * -----------------------------------------------------------------
 */

struct _CS_NVectorContent
{
  sunindextype length;          /* vector length       */
  sunbooleantype own_data;      /* data ownership flag */
  suncomplextype* complex_data; /* complex data array  */
};

typedef struct _CS_NVectorContent* CS_NVectorContent;

/*
 * -----------------------------------------------------------------
 * Macros CS_NV_CONTENT, CS_NV_DATA, CS_NV_OWN_DATA,
 *        CS_NV_LENGTH, and CS_NV_Ith
 * -----------------------------------------------------------------
 */

#define NV_CONTENT_CS(v) ((CS_NVectorContent)(v->content))

#define NV_LENGTH_CS(v) (NV_CONTENT_CS(v)->length)

#define NV_OWN_DATA_CS(v) (NV_CONTENT_CS(v)->own_data)

#define NV_COMPLEX_DATA_CS(v) (NV_CONTENT_CS(v)->complex_data)

#define NV_Ith_CS(v, i) (NV_COMPLEX_DATA_CS(v)[i])

/*
 * -----------------------------------------------------------------
 * Functions exported by nvector_serial
 * -----------------------------------------------------------------
 */

SUNDIALS_EXPORT
N_Vector N_VNewEmpty_SComplex(sunindextype vec_length, SUNContext sunctx);

SUNDIALS_EXPORT
N_Vector N_VNew_SComplex(sunindextype vec_length, SUNContext sunctx);

SUNDIALS_EXPORT
N_Vector N_VMake_SComplex(sunindextype vec_length, suncomplextype* v_data,
                      SUNContext sunctx);

SUNDIALS_EXPORT
sunindextype N_VGetLength_SComplex(N_Vector v);

SUNDIALS_EXPORT
void N_VPrint_SComplex(N_Vector v);

SUNDIALS_EXPORT
void N_VPrintFile_SComplex(N_Vector v, FILE* outfile);

static inline N_Vector_ID N_VGetVectorID_SComplex(N_Vector v)
{
  return SUNDIALS_NVEC_CUSTOM;
}

SUNDIALS_EXPORT
N_Vector N_VCloneEmpty_SComplex(N_Vector w);

SUNDIALS_EXPORT
N_Vector N_VClone_SComplex(N_Vector w);

SUNDIALS_EXPORT
void N_VDestroy_SComplex(N_Vector v);

SUNDIALS_EXPORT
void N_VSpace_SComplex(N_Vector v, sunindextype* lrw, sunindextype* liw);

SUNDIALS_EXPORT
suncomplextype* N_VGetArrayPointer_SComplex(N_Vector v);

SUNDIALS_EXPORT
void N_VSetArrayPointer_SComplex(suncomplextype* v_data, N_Vector v);

/* standard vector operations */
SUNDIALS_EXPORT
void N_VLinearSum_SComplex(suncomplextype a, N_Vector x, suncomplextype b, N_Vector y,
                       N_Vector z);

SUNDIALS_EXPORT
void N_VLinearSum_Real(sunrealtype a, N_Vector x, sunrealtype b, N_Vector y,
                       N_Vector z); 
               
SUNDIALS_EXPORT
void N_VConst_SComplex(suncomplextype c, N_Vector z);

SUNDIALS_EXPORT
void N_VConst_Real(sunrealtype c, N_Vector z);

SUNDIALS_EXPORT
void N_VProd_SComplex(N_Vector x, N_Vector y, N_Vector z);

SUNDIALS_EXPORT
void N_VDiv_SComplex(N_Vector x, N_Vector y, N_Vector z);

SUNDIALS_EXPORT
void N_VScale_SComplex(suncomplextype c, N_Vector x, N_Vector z);

SUNDIALS_EXPORT
void N_VScale_Real(sunrealtype c, N_Vector x, N_Vector z);

SUNDIALS_EXPORT
void N_VAbs_SComplex(N_Vector x, N_Vector z);

SUNDIALS_EXPORT
void N_VInv_SComplex(N_Vector x, N_Vector z);

SUNDIALS_EXPORT
void N_VAddConst_SComplex(N_Vector x, suncomplextype b, N_Vector z);

SUNDIALS_EXPORT
void N_VAddConst_Real(N_Vector x, sunrealtype b, N_Vector z);

SUNDIALS_EXPORT
suncomplextype N_VDotProd_SComplex(N_Vector x, N_Vector y);

SUNDIALS_EXPORT
sunrealtype N_VDotProd_Real(N_Vector x, N_Vector y);

SUNDIALS_EXPORT
sunrealtype N_VMaxNorm_SComplex(N_Vector x);

SUNDIALS_EXPORT
sunrealtype N_VWrmsNorm_SComplex(N_Vector x, N_Vector w);

SUNDIALS_EXPORT
sunrealtype N_VWL2Norm_SComplex(N_Vector x, N_Vector w);

SUNDIALS_EXPORT
sunrealtype N_VL1Norm_SComplex(N_Vector x);

/* fused vector operations */
SUNDIALS_EXPORT
SUNErrCode N_VLinearCombination_SComplex(int nvec, suncomplextype* c, N_Vector* V,
                                     N_Vector z);

SUNDIALS_EXPORT
SUNErrCode N_VLinearCombination_Real(int nvec, sunrealtype* c, N_Vector* V,
                                     N_Vector z);

SUNDIALS_EXPORT
SUNErrCode N_VScaleAddMulti_SComplex(int nvec, suncomplextype* a, N_Vector x,
                                 N_Vector* Y, N_Vector* Z);

SUNDIALS_EXPORT
SUNErrCode N_VScaleAddMulti_Real(int nvec, sunrealtype* a, N_Vector x,
                                 N_Vector* Y, N_Vector* Z);

SUNDIALS_EXPORT
SUNErrCode N_VDotProdMulti_SComplex(int nvec, N_Vector x, N_Vector* Y,
                                suncomplextype* dotprods);

SUNDIALS_EXPORT
SUNErrCode N_VDotProdMulti_Real(int nvec, N_Vector x, N_Vector* Y,
                                sunrealtype* dotprods);                                

/* vector array operations */
SUNDIALS_EXPORT
SUNErrCode N_VLinearSumVectorArray_SComplex(int nvec, suncomplextype a, N_Vector* X,
                                        suncomplextype b, N_Vector* Y,
                                        N_Vector* Z);

SUNDIALS_EXPORT
SUNErrCode N_VLinearSumVectorArray_Real(int nvec, sunrealtype a, N_Vector* X,
                                        sunrealtype b, N_Vector* Y,
                                        N_Vector* Z);

SUNDIALS_EXPORT
SUNErrCode N_VScaleVectorArray_SComplex(int nvec, suncomplextype* c, N_Vector* X,
                                    N_Vector* Z);

SUNDIALS_EXPORT
SUNErrCode N_VScaleVectorArray_Real(int nvec, sunrealtype* c, N_Vector* X,
                                    N_Vector* Z);

SUNDIALS_EXPORT
SUNErrCode N_VConstVectorArray_SComplex(int nvecs, suncomplextype c, N_Vector* Z);

SUNDIALS_EXPORT
SUNErrCode N_VConstVectorArray_Real(int nvecs, sunrealtype c, N_Vector* Z);

SUNDIALS_EXPORT
SUNErrCode N_VWrmsNormVectorArray_SComplex(int nvecs, N_Vector* X, N_Vector* W,
                                       sunrealtype* nrm);

SUNDIALS_EXPORT
SUNErrCode N_VScaleAddMultiVectorArray_SComplex(int nvec, int nsum,
                                            suncomplextype* a, N_Vector* X,
                                            N_Vector** Y, N_Vector** Z);

SUNDIALS_EXPORT
SUNErrCode N_VScaleAddMultiVectorArray_Real(int nvec, int nsum,
                                            sunrealtype* a, N_Vector* X,
                                            N_Vector** Y, N_Vector** Z);                                            

SUNDIALS_EXPORT
SUNErrCode N_VLinearCombinationVectorArray_SComplex(int nvec, int nsum,
                                                suncomplextype* c, N_Vector** X,
                                                N_Vector* Z);

SUNDIALS_EXPORT
SUNErrCode N_VLinearCombinationVectorArray_Real(int nvec, int nsum,
                                                sunrealtype* c, N_Vector** X,
                                                N_Vector* Z);                                                

/* OPTIONAL local reduction kernels (no parallel communication) */
SUNDIALS_EXPORT
sunrealtype N_VWSqrSumLocal_SComplex(N_Vector x, N_Vector w);

/* OPTIONAL XBraid interface operations */
SUNDIALS_EXPORT
SUNErrCode N_VBufSize_SComplex(N_Vector x, sunindextype* size);

SUNDIALS_EXPORT
SUNErrCode N_VBufPack_SComplex(N_Vector x, void* buf);

SUNDIALS_EXPORT
SUNErrCode N_VBufUnpack_SComplex(N_Vector x, void* buf);

#ifdef __cplusplus
}
#endif

#endif
