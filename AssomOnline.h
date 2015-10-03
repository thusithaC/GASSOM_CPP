#pragma once
#include "config.h"
#include <cmath>

/* Hard coded parameters in the model*/
#define SIGMA_N float(0.2) //0.2
#define SIGMA_W float(2) //2
#define INPUTSQNORM 1


struct Params
{
	int TOPO_SUBSPACE;
	int BASESNUM;
	int BASISDIM;
	int AGENTSNUM;
	float ALPHA_A;
	float ALPHA_C;
	float SIGMA_A;
	float SIGMA_C;
	float TCONST;
	float TCONST2;

};



typedef struct Params ASParams;

class AssomOnline
{
private:



	float* coef;
	float* residue;
	float* corrBX1;
	float* corrBX2;
	float* corrTmp1;
	float* corrTmp2;
	float* probTmp1;
	int *index;
	float* proj;
	float* emission;
	float* weightedBases;
	float* transProb;
	float *residueHost;
	float *winProj;
	float *winProjHost;
	
	
	int* winners;
	float *wInputTmp1;
	float *wInputTmp2;
	float *wInputTmp3;
	float *wInputTmp4;
	float *diff1;
	float *diff2;
	float *tmpBases1;
	float *tmpBases2;
	float *winErr;
	int iter;
	float normSigmaW;
	float normSigmaN;
	float *ones;

	void cudaAssomEncode();
	void cudaNormalizeBases();
	void cudaNormalizeProb();
	void cudaUpdateBases();

public:
	int TOPO_SUBSPACE;
	int BASESNUM;
	int BASISDIM;
	int AGENTSNUM;
	float ALPHA_A;
	float ALPHA_C;
	float SIGMA_A;
	float SIGMA_C;
	float TCONST;
	float TCONST2;
	
	float* bases1;
	float* bases2;
	float* nodeProb;


public:
	AssomOnline(){}
	~AssomOnline();
	AssomOnline(float* batchInput, ASParams p);
	void AssomEncode();
	void updateBases();
	float* getBases();
	float* getCoef();
	float* getResidue();
	float* getResidueHost();
	float getErr();
	void hostGenTransitionProb(float* hostTransitionProb, float sigma, float alpha);
	void setBases(float*hostBases1, float*hostBases2);
	float* getWinProjHost();
	void preprocess(int flag);
	float* getProj();
};
