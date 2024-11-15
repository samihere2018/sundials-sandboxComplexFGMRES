/* -----------------------------------------------------------------
 * Programmer(s): Daniel Reynolds and Sylvia Amihere @ SMU
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
 * These test functions check some components of a complex-valued
 * SUNLINEARSOLVER module implementation (for more thorough tests,
 * see the main SUNDIALS repository, inside examples/sunlinsol/).
 * -----------------------------------------------------------------
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "nvector_serialcomplex.h"
#include "sundials_iterativecomplex.h"
// #include "sundials_iterativecomplex.h" //Amihere
// #include "sundials_iterativecomplex_impl.h" //Amihere


#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif

SUNErrCode SUNClassicalGSComplex(N_Vector*, suncomplextype**, int k, int p,
                          suncomplextype* new_vk_norm, suncomplextype* stemp,
                          N_Vector* vtemp);

SUNErrCode SUNModifiedGSComplex(N_Vector* v, suncomplextype** h, int k, int p,
                         suncomplextype* new_vk_norm);

int SUNQRfactComplex(int n, suncomplextype** h, suncomplextype* q, int job);

int main(int argc, char* argv[])
{
  N_Vector* V;
  N_Vector* VNew;
  N_Vector x;
  N_Vector* vtemp;
  suncomplextype* stemp;
  suncomplextype* vdata;
  suncomplextype* givens;
  SUNContext sunctx;
  suncomplextype** H;
  int k, l, job, n, krydim;
  suncomplextype vnorm, ssnorm;

  if (SUNContext_Create(SUN_COMM_NULL, &sunctx))
  {
    printf("ERROR: SUNContext_Create failed\n");
    return (-1);
  }

  /* Create vectors */
  x = N_VNew_SComplex(3, sunctx);
  V = N_VCloneVectorArray(3, x);
  VNew = N_VCloneVectorArray(3, x);

  // H = NULL;
  // if (H == NULL)
  // {
  //   H = (suncomplextype**)malloc((3 + 1) * sizeof(suncomplextype*));

  //   for (k = 0; k <= 3; k++)
  //   {
  //     H[k] = NULL;
  //     H[k] = (suncomplextype*)malloc(3 * sizeof(suncomplextype));
  //   }
  // }

  H = (suncomplextype**)malloc((3) * sizeof(suncomplextype*)); //Amihere
  for (k = 0; k < 3; k++)
  {
    H[k] = NULL;
    H[k] = (suncomplextype*)malloc(3 * sizeof(suncomplextype)); //Amihere
  }

  givens = (suncomplextype*)malloc(2 * 3 * sizeof(suncomplextype));

  vtemp = (N_Vector*)malloc((3) * sizeof(N_Vector)); //Amihere
  stemp = (suncomplextype*)malloc((3) * sizeof(suncomplextype)); //Amihere

  /* set up matrix */
  vdata = N_VGetArrayPointer_SComplex(V[0]);
  vdata[0] = 3.0 + 4.0*I;
  vdata[1] = 0.0;
  vdata[2] = 0.0;
  vdata = N_VGetArrayPointer_SComplex(V[1]);
  vdata[0] = -4.0+3.0*I;
  vdata[1] = 8.0+6.0*I;
  vdata[2] = 0.0;
  vdata = N_VGetArrayPointer_SComplex(V[2]);
  vdata[0] = 4.0-3.0*I;
  vdata[1] = 12.0+9.0*I;
  vdata[2] = 1.0;

  /* perform Gram-Schmidt process for all vectors in V */
  for (k=0; k<3; k++)
  {
    krydim = k  ;
    SUNClassicalGSComplex(V, H, k, 3, &vnorm, stemp, vtemp);
    // SUNModifiedGSComplex(V, H, k, 3, &vnorm);
    N_VScale_SComplex(1.0/vnorm, V[k], V[k]);
    SUNQRfactComplex(krydim, H, givens, k);
    // printf("stemp at %d is: %f + i%f\n", k, creal(stemp[k]), cimag(stemp[k]));
    // N_VScale_SComplex(1.0/vnorm, V[k], VNew[k]);
  }
   
  /* check dot product results */
  for (k=0; k<3; k++)
  {
    for (l=0; l<3; l++)
    {
        vnorm = N_VDotProd_SComplex(V[k],V[l]);
        printf("<V[%i],V[%i]> = %e + %e I\n", l, k, creal(vnorm), cimag(vnorm));
    }
  }
  
  /* print everything in V */
  for (k=0; k<3; k++)
  {
    printf("V[%i] = \n",k);
    N_VPrint(V[k]);
  }


  /* print everything in H */
  for (k=0; k<3; k++)
  {
    for (l=0; l<3; l++) {
      printf("H at row %d, column %d is: %f + i%f\n", k, l, creal(H[k][l]), cimag(H[k][l]));
    }
  }

  /* return with success */
  return 0;
}
