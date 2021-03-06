/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

/*
   Common operations for ComputeNonbonded classes
*/

#include "ComputeNonbondedInl.h"

// 3 inline functions to handle explicit calculation of separation-shifted
// vdW for FEP and TI, and a shifted electrostatics potential for decoupling

/* ******************************************** */
/* vdW energy, force and lambda2 energy for FEP */
/* ******************************************** */
inline void fep_vdw_forceandenergies (BigReal A, BigReal B, BigReal r2, 
  BigReal myVdwShift, BigReal myVdwShift2, BigReal switchdist2, BigReal cutoff2, 
  BigReal myVdwLambda, BigReal myVdwLambda2, Bool Fep_WCA_repuOn, Bool Fep_WCA_dispOn, bool Fep_Wham, 
  BigReal WCA_rcut1, BigReal WCA_rcut2, BigReal WCA_rcut3, BigReal switchfactor, 
  BigReal* alch_vdw_energy, BigReal* alch_vdw_force, 
  BigReal* alch_vdw_energy_2, BigReal* alch_vdw_energy_2_Left) {

  // switching function (this is correct whether switching is active or not)
  const BigReal switchmul = r2 > switchdist2? \
           switchfactor*(cutoff2 - r2)*(cutoff2 - r2)*(cutoff2 - 3.*switchdist2 + 2.*r2) \
           : 1.;
  const BigReal switchmul2 = (r2 > switchdist2)? \
           12.*switchfactor*(cutoff2 - r2)*(r2 - switchdist2) : 0.;
  
  *alch_vdw_energy_2_Left = 0.;

  if(Fep_WCA_repuOn){
    const BigReal Emin = B*B/(4.0*A);
    const BigReal Rmin_SQ = powf(2.0*A/B, 1.f/3);
    const BigReal r2_1 = r2 + (1.-WCA_rcut1)*(1.-WCA_rcut1)*Rmin_SQ;
    const BigReal r2_2 = r2 + (1.-WCA_rcut2)*(1.-WCA_rcut2)*Rmin_SQ;
    const BigReal r2_3 = r2 + (1.-WCA_rcut3)*(1.-WCA_rcut3)*Rmin_SQ;
    const BigReal r6_1 = r2_1*r2_1*r2_1;
    const BigReal r6_2 = r2_2*r2_2*r2_2;
    const BigReal r6_3 = r2_3*r2_3*r2_3;
    const BigReal WCA_rcut1_energy = r2 <= Rmin_SQ * \
                                     (1.0 - (1.0-WCA_rcut1)*(1.0-WCA_rcut1))? \
                                     A/(r6_1*r6_1) - B/r6_1 + Emin : 0.;
    const BigReal WCA_rcut2_energy = r2 <= Rmin_SQ * \
                                     (1.0 - (1.0-WCA_rcut2)*(1.0-WCA_rcut2))? \
                                     A/(r6_2*r6_2) - B/r6_2 + Emin : 0.;
    const BigReal WCA_rcut3_energy = r2 <= Rmin_SQ * \
                                     (1.0 - (1.0-WCA_rcut3)*(1.0-WCA_rcut3))? \
                                     A/(r6_3*r6_3) - B/r6_3 + Emin : 0.;
    const BigReal WCA_rcut1_force = r2 <= Rmin_SQ * \
                                    (1.0 - (1.0-WCA_rcut1)*(1.0-WCA_rcut1))? \
                                    (12.*(WCA_rcut1_energy) \
                                     + 6.*B/r6_1 - 12.0 * Emin )/r2_1: 0.;
    const BigReal WCA_rcut2_force = r2 <= Rmin_SQ * \
                                    (1.0 - (1.0-WCA_rcut2)*(1.0-WCA_rcut2))? \
                                    (12.*(WCA_rcut2_energy) \
                                     + 6.*B/r6_2 - 12.0 * Emin )/r2_2: 0.;
    const BigReal WCA_rcut3_force = r2 <= Rmin_SQ * \
                                    (1.0 - (1.0-WCA_rcut3)*(1.0-WCA_rcut3))? \
                                    (12.*(WCA_rcut3_energy) \
                                     + 6.*B/r6_3 - 12.0 * Emin )/r2_3: 0.;
  // separation-shifted repulsion force and energy
    *alch_vdw_energy = WCA_rcut2_energy; 
    *alch_vdw_force =  WCA_rcut2_force;
    if(WCA_rcut1 < WCA_rcut2) {
      *alch_vdw_energy_2_Left = *alch_vdw_energy + WCA_rcut2_energy - WCA_rcut1_energy; 
    }
    if(WCA_rcut2 < WCA_rcut3) {
      *alch_vdw_energy_2 = *alch_vdw_energy + WCA_rcut3_energy - WCA_rcut2_energy; 
    }
  }
  else if(Fep_WCA_dispOn){
    const BigReal Emin = B*B/(4.0*A);
    const BigReal Rmin_SQ = powf(2.0*A/B, 1.f/3);
    const BigReal r2_1 = r2;
    const BigReal r2_2 = r2;
    const BigReal r6_1 = r2_1*r2_1*r2_1;
    const BigReal r6_2 = r2_2*r2_2*r2_2;
  // separation-shifted dispersion force and energy
    if(B == 0.0)	{	// some atoms like H, drude, lone pair are involved
			*alch_vdw_energy = 0.0;
			*alch_vdw_force = 0.0;
			*alch_vdw_energy_2 = 0.0;
    }
    else {
      *alch_vdw_energy = r2 > Rmin_SQ? \
                         myVdwLambda*(A/(r6_1*r6_1) - B/r6_1): \
                         A/(r6_1*r6_1) - B/r6_1 + (1.-myVdwLambda)* Emin;
      *alch_vdw_force =  r2 > Rmin_SQ? \
                         (12.*(*alch_vdw_energy) + 6.*myVdwLambda*B/r6_1)/r2_1 * switchmul \
                         + (*alch_vdw_energy) * switchmul2:(12.*(*alch_vdw_energy) \
                         + 6.*B/r6_1 - 12.0*(1.-myVdwLambda)* Emin )/r2_1;
      *alch_vdw_energy *= switchmul; 
			if(!Fep_Wham){ 
        *alch_vdw_energy_2 = r2 > Rmin_SQ? \
                             myVdwLambda2*switchmul*(A/(r6_1*r6_1) - B/r6_1): \
                             A/(r6_1*r6_1) - B/r6_1 + (1.-myVdwLambda2)* Emin;
			}
			else{
				*alch_vdw_energy_2 = r2 > Rmin_SQ? \
					                   switchmul*(A/(r6_1*r6_1) - B/r6_1): - Emin;
        *alch_vdw_energy_2 += *alch_vdw_energy;
			}
    }
  } 
  else {
    const BigReal r2_1 = r2 + myVdwShift;  //myVdwShift already multplied by relevant (1-vdwLambda)
    const BigReal r2_2 = r2 + myVdwShift2;
    const BigReal r6_1 = r2_1*r2_1*r2_1;
    const BigReal r6_2 = r2_2*r2_2*r2_2;
  // separation-shifted vdW force and energy
    *alch_vdw_energy = A/(r6_1*r6_1) - B/r6_1;
    *alch_vdw_force =  myVdwLambda * (  \
        + (12.*(*alch_vdw_energy) + 6.*B/r6_1)/r2_1 * switchmul \
        + (*alch_vdw_energy) * switchmul2);
    *alch_vdw_energy *= myVdwLambda*switchmul;
    *alch_vdw_energy_2 = (A/(r6_2*r6_2) - B/r6_2) * myVdwLambda2 * switchmul;
  }

}

#define FEPFLAG
#define CALCENERGY

#define NBTYPE NBPAIR
#include "ComputeNonbondedBase.h"
#define FULLELECT
#include "ComputeNonbondedBase.h"
#define MERGEELECT
#include "ComputeNonbondedBase.h"
#undef MERGEELECT
#define SLOWONLY
#include "ComputeNonbondedBase.h"
#undef SLOWONLY
#undef FULLELECT
#undef  NBTYPE

#define NBTYPE NBSELF
#include "ComputeNonbondedBase.h"
#define FULLELECT
#include "ComputeNonbondedBase.h"
#define MERGEELECT
#include "ComputeNonbondedBase.h"
#undef MERGEELECT
#define SLOWONLY
#include "ComputeNonbondedBase.h"
#undef SLOWONLY
#undef FULLELECT
#undef  NBTYPE

#undef CALCENERGY
#undef FEPFLAG


