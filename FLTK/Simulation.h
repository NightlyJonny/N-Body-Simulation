#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <signal.h>
#include <string>
#include <cstring>
#include <thread>
#include <chrono>
#include "Particle.h"
#include "Vector2.h"
#define FRAMERATE 60
#define EFACTOR 0.2
#define KFACTOR 0 // 0.01
using namespace std;

class Simulation {

private:
	ofstream outFile;
	Particle* particles;
	int DURATION, FRAMESTEP, NPARTICLE;
	int frames = 0;
	bool pause = false;
	const int targetDt = 1000000 / FRAMERATE;

public:
	Simulation(int argc, char** argv);
	void core();
	void saveFrame(ofstream& outFile, Particle* particles, int particleNumber);
	void saveProgress(char* ofName, Particle* particles, int particleNumber, int frameStep);
	double random(double min, double max);
	void printProgress(int currentFrame, int totalFrames);
	void integrator(Particle* particles, int NPARTICLE, int FRAMESTEP, int N, double* coefc, double* coefd);
	Particle* getParticle() const;
	int getNParticle() const;
	int* getFrameRef();
	void setPause(bool);
	void togglePause();
	void stop();
	void terminateSim();
};