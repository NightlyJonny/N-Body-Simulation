#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <signal.h>
#include <string>
#include <cstring>
#include <thread>
#include <chrono>
#include <climits>
#include "Particle.h"
#include "Vector3.h"
#define FRAMERATE 60
#define EFACTOR 0.2
#define KFACTOR 0 // 0.01
#define ZEROTHRESHOLD 0.2
#define DENSITY 0.2
#define PI 3.14159265359
using namespace std;

class Simulation {

private:
	fstream outFile;
	Particle* particles;
	unsigned int DURATION = UINT_MAX, FRAMESTEP = 10, NPARTICLE = 200;
	int frames = 0;
	bool pause = false;
	bool frameLimit;
	const int targetDt = 1000000 / FRAMERATE;

public:
	Simulation(string outFile, unsigned int duration, unsigned int frameStep, unsigned int particleNumber, bool frameLimit, unsigned long int seed);
	void core();
	void saveFrame(fstream& outFile, Particle* particles, int particleNumber);
	void saveProgress(char* ofName, Particle* particles, int particleNumber, int frameStep);
	float random(float min, float max);
	void printProgress(int currentFrame, int totalFrames);
	void integrator(Particle* particles, int NPARTICLE, int FRAMESTEP, int N, float* coefc, float* coefd);
	Particle* getParticle() const;
	int getNParticle() const;
	int* getFrameRef();
	void setPause(bool);
	void togglePause();
	void stop();
	void terminateSim();
	unsigned int getDuration() { return (DURATION < UINT_MAX ? DURATION : 0); }
	string debugText = "";
};