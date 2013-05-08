/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#ifndef COMPUTEPME_H
#define COMPUTEPME_H

#include "Compute.h"
#include "PmeBase.h"
#include "NamdTypes.h"
#include "PatchTypes.h"
#include "Box.h"
#include "OwnerBox.h"

class PmeRealSpace;
class ComputeMgr;
class SubmitReduction;
class PmeGridMsg;
class ComputePmeMgr;
class Patch;

class ComputePme : public Compute {
public:
  ComputePme(ComputeID c, PatchID pid);
  virtual ~ComputePme();
  void initialize();
  int noWork();
  void doWork();
  void ungridForces();
  void setMgr(ComputePmeMgr *mgr) { myMgr = mgr; }

 private:
  PatchID patchID;
  Patch *patch;
  Box<Patch,CompAtom> *positionBox;
  Box<Patch,CompAtom> *avgPositionBox;
  Box<Patch,Results> *forceBox;

  PmeGrid myGrid;
  int alchFepOn, alchThermIntOn, lesOn, lesFactor, pairOn, selfOn, numGrids;
  int alchDecouple;
  BigReal alchElecLambdaStart;
  BigReal elecLambdaUp;
  BigReal elecLambdaDown;
  
  PmeRealSpace *myRealSpace[PME_MAX_EVALS];
  int numLocalAtoms;
  PmeParticle *localData;
  ResizeArray<PmeParticle> localData_alloc;
  unsigned char *localPartition;
  ResizeArray<unsigned char> localPartition_alloc;
  ResizeArray<Vector> localResults_alloc;
  int numGridAtoms[PME_MAX_EVALS];
  PmeParticle *localGridData[PME_MAX_EVALS];
  ComputePmeMgr *myMgr;

};

#endif
