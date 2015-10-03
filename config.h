#pragma once


#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>
#include <cublas_v2.h>
#include <opencv2\imgproc\imgproc.hpp>
#include <string>
#include <time.h>
#include <stdio.h>
#include <yarp/os/Network.h>
#include <yarp/dev/ControlBoardInterfaces.h>
#include <yarp/dev/all.h>
#include <yarp/os/Time.h>
#include <yarp/sig/all.h>

using namespace yarp::dev;
using namespace yarp::sig;
using namespace yarp::os;
using namespace std;
using namespace cv;

#define CV_TYPE CV_32FC1
#define EPS	1E-16
//#define T_TRAIN	int(2E4) defined in Icub_vergence.h
#define PI 3.14159265358979325
#define CHECKPOINTSNUM 20
#define BLASSTREAMS 2

#define TESTTRAJ 0
#define NOVERGENCE 0

//#define TOPO_SUBSPACE 18
//#define BASESNUM int(TOPO_SUBSPACE*TOPO_SUBSPACE)
//#define BASISDIM 200
//#define AGENTSNUM 100
//#define ALPHA_A float(0.0050) //0.005
//#define ALPHA_C float(0.0005)
//#define SIGMA_A float(2.5)
//#define SIGMA_C float(0.1)
//#define TCONST float(40000) //float(50000)
//#define TCONST2 float(20000) 
#define MAXAGENTS 100




#define FOVEAWIDTH 55
#define PATCHWIDTH 10
#define PATCHSHIFT 5
#define PATCHNUM AGENTSNUM
#define PUPILDIST 120//300nisl
#define ROWPATCHNUM 10 //sqrt<int>(AGENTSNUM)
#define FOV 5.5/180*PI

#define FEATUREDIM int(1+BASESNUM)

#define IMGPLANE float(0.8) //0.8

#define BLOCKSIZE 128
#define BLOCKSIZE2 224
#define BLOCKSIZE3 256
#define BLOCKSIZE4 400


extern cublasHandle_t cuhandle;
extern cublasHandle_t cuhandles[];
extern cudaStream_t streams[];
extern float cublasOne;
extern float cublasZero;
extern float cublasNegOne;
using namespace cv;

//#pragma comment(lib, "cudart") 
#pragma comment(lib, "cublas")
//#pragma comment(lib, "curand")
#pragma comment(lib, "libeng")
#pragma comment(lib, "libmwlapack")
#pragma comment(lib, "libmex")
#pragma comment(lib, "libmx")


#ifdef NDEBUG  
	#pragma comment(lib, "icubmod")
	#pragma comment(lib, "gazecontrollerclient")
	#pragma comment(lib, "cartesiancontrollerclient")
	#pragma comment(lib, "controlboardwrapper2")
	#pragma comment(lib, "cartesiancontrollerserver")
	#pragma comment(lib, "actionPrimitives")
	#pragma comment(lib, "canLoaderLib")
	#pragma comment(lib, "ctrlLib")
	#pragma comment(lib, "debugStream")
	#pragma comment(lib, "iCubDev")
	#pragma comment(lib, "iCubTestLib")
	#pragma comment(lib, "iCubVis")
	#pragma comment(lib, "iDyn")
	#pragma comment(lib, "iKin")
	#pragma comment(lib, "perceptiveModels")
	#pragma comment(lib, "skinDynLib")
	#pragma comment(lib, "YARP_OS")
	#pragma comment(lib, "YARP_sig")
	#pragma comment(lib, "YARP_serversql")
	#pragma comment(lib, "YARP_math")
	#pragma comment(lib, "YARP_dev")
	#pragma comment(lib, "yarpcar")
	#pragma comment(lib, "YARP_name")
	#pragma comment(lib, "YARP_init")
	#pragma comment(lib, "YARP_bayer")
	#pragma comment(lib, "gsl")
	#pragma comment(lib, "gslcblas")
	#pragma comment(lib, "winmm")
	#pragma comment(lib, "ACE")
	#pragma comment(lib, "opencv_core247")
	#pragma comment(lib, "opencv_highgui247")
	#pragma comment(lib, "opencv_imgproc247")
	#pragma comment(lib, "cuFFT")


#endif

#ifdef _DEBUG
	#pragma comment(lib, "icubmodd")
	#pragma comment(lib, "gazecontrollerclientd")
	#pragma comment(lib, "cartesiancontrollerclientd")
	#pragma comment(lib, "controlboardwrapper2d")
	#pragma comment(lib, "cartesiancontrollerserverd")
	#pragma comment(lib, "actionPrimitivesd")
	#pragma comment(lib, "canLoaderLibd")
	#pragma comment(lib, "ctrlLibd")
	#pragma comment(lib, "debugStreamd")
	#pragma comment(lib, "iCubDevd")
	#pragma comment(lib, "iCubTestLibd")
	#pragma comment(lib, "iCubVisd")
	#pragma comment(lib, "iDynd")
	#pragma comment(lib, "iKind")
	#pragma comment(lib, "perceptiveModelsd")
	#pragma comment(lib, "skinDynLibd")
	#pragma comment(lib, "YARP_OSd")
	#pragma comment(lib, "YARP_sigd")
	#pragma comment(lib, "YARP_serversqld")
	#pragma comment(lib, "YARP_mathd")
	#pragma comment(lib, "YARP_devd")
	#pragma comment(lib, "yarpcard")
	#pragma comment(lib, "YARP_named")
	#pragma comment(lib, "YARP_initd")
	#pragma comment(lib, "YARP_bayerd")
	#pragma comment(lib, "gsl")
	#pragma comment(lib, "gslcblas")
	#pragma comment(lib, "winmm")
	#pragma comment(lib, "ACEd")
	#pragma comment(lib, "opencv_core247d")
	#pragma comment(lib, "opencv_highgui247d")
	#pragma comment(lib, "opencv_imgproc247d")
	#pragma comment(lib, "cuFFT")
	#pragma message("Please ignore LNK4099, it is not an issue.") 
#endif

