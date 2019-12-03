#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <signal.h>
#include <string>
#include <cstring>
#include <climits>
#include "Particle.h"
#include "Vector2.h"
#define FRAMERATE 60
#define EFACTOR 0.2
#define KFACTOR 0 // 0.01
#define ZEROTHRESHOLD 0.1
using namespace std;

void saveFrame (ofstream& outFile, Particle* particles, int particleNumber) {
	for (int i = 0; i < particleNumber; i++) {
		if (particles[i].active)
			outFile << particles[i].radius << ":" << particles[i].position.x << "," << particles[i].position.y << ";" << particles[i].angle << "\t";
		else
			outFile << "0" << "\t";
	}
	outFile << "\n";
}

void saveProgress (char* ofName, Particle* particles, int particleNumber, int frameStep) {
	ofstream outFile ((string(ofName) + string(".dat")).c_str(), ios::out | ios::trunc | ios::binary);
	if (!outFile.is_open()) {
		cerr << "Error opening output file." << endl;
		return;
	}

	outFile.write((char*)(&particleNumber), sizeof(particleNumber));
	outFile.write((char*)(&frameStep), sizeof(frameStep));
	for (int i = 0; i < particleNumber; i++) {
		outFile.write((char*)(particles + i), sizeof(particles[i]));
	}

	outFile.close();
}

double random (double min, double max) {

	return ((double)rand() / RAND_MAX) * (max-min) + min;
}

void printProgress (int currentFrame, int totalFrames) {
	if (totalFrames > 0) {
		double percentage = ((double)currentFrame / (double)totalFrames) * 100;
		cout << "\r[";
		for (int i = 0; i < 50; i++) {
			if (i == 24) cout << round(percentage) << " %";
			else if (i < round(percentage/2.0)) cout << "#";
			else cout << " ";
		}
		cout << "]" << flush;
	}
	else {
		cout << fixed << setprecision(1) << "\rGenerated " << (currentFrame/FRAMERATE) << " seconds.\t" << flush;
	}
}

bool stopped = false;
void sigHandler (int sig) {
	cout << flush << "\nRendering stopped, saving progress..." << endl;
	stopped = true;
}

void integrator (Particle* particles, int NPARTICLE, int FRAMESTEP, int N, double* coefc, double* coefd) {
	Vector2* force = new Vector2[NPARTICLE];
	const double t = 1.0/(FRAMERATE*FRAMESTEP);

	for (int n = 0; n < N; n ++) {
		for (int p1 = 0; p1 < NPARTICLE; p1++) {
			if (!particles[p1].active) continue;
			for (int p2 = p1+1; p2 < NPARTICLE; p2++) {
				if (!particles[p2].active) continue;
				Vector2 dVector = particles[p2].position - particles[p1].position;
				double distance = dVector.norm();
				Vector2 curForce = dVector.versor() * particles[p1].mass * particles[p2].mass / pow(distance, 2);
				force[p1] = force[p1] + curForce;
				force[p2] = force[p2] - curForce;
			}
		}
		for (int p = 0; p < NPARTICLE; ++p) {
			if (!particles[p].active) continue;
			particles[p].velocityStep(coefc[n] * t, force[p]);
			particles[p].positionStep(coefd[n] * t, force[p]);
			particles[p].angularStep(t/N);

			force[p] = particles[p].position * -KFACTOR;
		}
	}

	delete[] force;
}

int main(int argc, char** argv) {
	signal(SIGINT, sigHandler);
	char* outFileName;
	unsigned int DURATION = 60, FRAMESTEP = 10, NPARTICLE = 100; // Optional terminal paramenters IN THIS ORDER!
	bool forever = false;
	ofstream outFile;
	Particle* particles;
	double k0 = 0;
	{
		if (argc > 1) outFileName = argv[argc-1];
		else {
			cerr << "You must specify an output or save file.\nUsage: ./core [DURATION] [FRAMESTEP] [NPARTICLE] \"OutputFile.txt\"\nor\n./core \"ProgressData.dat\"" << endl;
			return 1;
		}
		bool resuming = string(outFileName).substr(string(outFileName).length()-4, 4).compare(string(".dat")) == 0;
		if (!resuming) {
			if (argc > 2) NPARTICLE = stoi(argv[argc-2]);
			if (argc > 3) FRAMESTEP = stoi(argv[argc-3]);
			if (argc > 4) DURATION = stoi(argv[argc-4]);
		}
		else {
			if (argc > 2) DURATION = stoi(argv[argc-2]);
		}
		if (DURATION == 0) {
			DURATION = UINT_MAX;
			forever = true; // Gem, Ensi, Bassi.
		}

		particles = new Particle[NPARTICLE];
		if (resuming) {
			ifstream inFile (outFileName, ios::in | ios::binary);
			if (!inFile.is_open()) {
				cerr << "Error opening input file." << endl;
				return 2;
			}

			inFile.read((char*)&NPARTICLE, sizeof(NPARTICLE));
			inFile.read((char*)&FRAMESTEP, sizeof(FRAMESTEP));
			for (int p = 0; p < NPARTICLE; p++) {
				inFile.read((char*)(particles+p), sizeof(particles[p]));
				if (inFile.tellg() == -1) {
					cerr << "Error reading from save file" << endl;
					return 2;
				}
			}
			inFile.close();
			strcpy(outFileName, string(outFileName).substr(0, string(outFileName).length()-4).c_str());
		}
		else {
			for (int p = 0; p < NPARTICLE; p++) {
				particles[p].initialize();
			}
		}

		outFile.open(outFileName, ios::out | (resuming ? ios::app : ios::trunc));
		if (!outFile.is_open()) {
			cerr << "Error opening output file." << endl;
			return 2;
		}
		outFile << NPARTICLE << "\n";
	}

	// Second order
	// double cs[] = {0, 1};
	// double cd[] = {0.5, 0.5};

	// Third order
	// double cs[] = {7.0/24, 3.0/4, -1.0/24};
	// double cd[] = {2.0/3, -2.0/3, 1};

	// Other fourth order
	double cs[] = {0, 1.351207191959657, -1.702414383919315, 1.351207191959657};
	double cd[] = {0.675603595979828, -0.175603595979828, -0.175603595979828, 0.675603595979828};

	int frames = 0;
	while ((frames < DURATION*FRAMERATE) && (!stopped)) {
		for (int i = 0; i < FRAMESTEP; i ++) {

			// Particle movement
			integrator (particles, NPARTICLE, FRAMESTEP, 4, cs, cd);

			// Collision detection (discrete)
			for (int p1 = 0; p1 < NPARTICLE; p1++) {
				if (!particles[p1].active) continue;
				for (int p2 = p1 + 1; p2 < NPARTICLE; p2++) {
					if (!particles[p2].active) continue;
					Vector2 dVector = particles[p1].position - particles[p2].position;
					Vector2 nVersor = dVector.versor();
					if (dVector.norm() <= particles[p1].radius + particles[p2].radius) {

						// Collision response
						Vector2 cVector = nVersor * 1.0001 * (particles[p1].radius + particles[p2].radius) - dVector;
						particles[p1].position = particles[p1].position + cVector * (particles[p1].mass / (particles[p1].mass + particles[p2].mass));
						particles[p2].position = particles[p2].position - cVector * (particles[p2].mass / (particles[p1].mass + particles[p2].mass));

						Vector2 vrVector = particles[p1].velocity - particles[p2].velocity;
						double nvr = vrVector * nVersor;
						if (abs(nvr) < ZEROTHRESHOLD) {
							Vector2 nVector = nVersor * particles[p1].radius;
							particles[p1].position = (particles[p1].position*particles[p1].mass + particles[p2].position*particles[p2].mass) / (particles[p1].mass + particles[p2].mass);
							particles[p1].velocity = (particles[p1].velocity*particles[p1].mass + particles[p2].velocity*particles[p2].mass) / (particles[p1].mass + particles[p2].mass);
							double newRadPow2 = particles[p1].radius*particles[p1].radius + particles[p2].radius*particles[p2].radius;
							particles[p1].omega = (particles[p1].mass * particles[p1].radius*particles[p1].radius * particles[p1].omega + particles[p2].mass * particles[p2].radius*particles[p2].radius * particles[p2].omega + 2 * particles[p2].mass * (nVector.x*vrVector.y - vrVector.x*nVector.y)) / ((particles[p1].mass + particles[p2].mass) * newRadPow2);
							particles[p1].mass += particles[p2].mass;
							particles[p1].radius = sqrt(newRadPow2);
							
							particles[p2].active = false;
						}
						else {
							double Jr = ((EFACTOR + 1) / (1 / particles[p1].mass + 1 / particles[p2].mass)) * nvr;
							particles[p1].velocity = particles[p1].velocity - nVersor * (Jr / particles[p1].mass);
							particles[p2].velocity = particles[p2].velocity + nVersor * (Jr / particles[p2].mass);
						}
					}
				}
			}
		}
		saveFrame(outFile, particles, NPARTICLE);
		frames ++;
		if (!stopped) {
			if (!forever) printProgress(frames, DURATION*FRAMERATE);
			else printProgress(frames, 0);
		}
	}
	cout << "\n";
	outFile.close();
	// saveProgress(outFileName, particles, NPARTICLE, FRAMESTEP); // NOT WORKING RN!
	delete[] particles;
	return 0;
} 