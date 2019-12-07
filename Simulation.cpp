#include "Simulation.h"

bool stopped = false;
void sigHandler(int sig) {
	cout << flush << "\nRendering stopped, saving progress..." << endl;
	stopped = true;
}

Simulation::Simulation(string outFN, unsigned int duration, unsigned int frameStep, unsigned int particleNumber, bool fl) {
	signal(SIGINT, sigHandler);
	srand(time(0));

	frameLimit = fl;
	string outFileName = outFN;
	DURATION = duration;
	if (DURATION == 0) DURATION = UINT_MAX;
	FRAMESTEP = frameStep;
	NPARTICLE = particleNumber;

	bool resuming = outFileName.substr(outFileName.length() - 4, 4).compare(string(".dat")) == 0;
	particles = new Particle[NPARTICLE];
	if (resuming) {
		ifstream inFile(outFileName.c_str(), ios::in | ios::binary);
		if (!inFile.is_open()) {
			cerr << "Error opening input file." << endl;
			exit(2);
		}

		inFile.read((char*)&NPARTICLE, sizeof(NPARTICLE));
		inFile.read((char*)&FRAMESTEP, sizeof(FRAMESTEP));
		for (int p = 0; p < NPARTICLE; p++) {
			inFile.read((char*)(particles + p), sizeof(particles[p]));
			if (inFile.tellg() == -1) {
				cerr << "Error reading from save file" << endl;
				exit(2);
			}
		}
		inFile.close();
		outFileName = outFileName.substr(0, outFileName.length() - 4);
	}
	else {
		for (int p = 0; p < NPARTICLE; p++) {
			particles[p].initialize();
		}
	}

	outFile.open(outFileName.c_str(), ios::out | (resuming ? ios::app : ios::trunc));
	if (!outFile.is_open()) {
		cerr << "Error opening output file." << endl;
		return;
	}
	outFile << NPARTICLE << "\n";
}

void Simulation::core() {

	// Second order
	// double cs[] = {0, 1};
	// double cd[] = {0.5, 0.5};

	// Third order
	// double cs[] = {7.0/24, 3.0/4, -1.0/24};
	// double cd[] = {2.0/3, -2.0/3, 1};
	
	// Fourth order
	double cs[] = {0, 1.351207191959657, -1.702414383919315, 1.351207191959657};
	double cd[] = {0.675603595979828, -0.175603595979828, -0.175603595979828, 0.675603595979828};

	while ((frames < DURATION * FRAMERATE) && !stopped) {
		auto start = chrono::high_resolution_clock::now();
		for (int i = 0; i < FRAMESTEP; i++) {

			// Particle movement
			integrator(particles, NPARTICLE, FRAMESTEP, 4, cs, cd);
			
			//It checks if the simulation is in pause, wait for 0.01 second to recheck the variable
			if (frameLimit) {
				while(pause && !stopped) { this_thread::sleep_for(chrono::milliseconds(10)); }
			}

			// Collision detection (discrete)
			for (int p1 = 0; p1 < NPARTICLE; p1++) {
				if (!particles[p1].active) continue;
				for (int p2 = p1 + 1; p2 < NPARTICLE; p2++) {
					if (!particles[p2].active) continue;
					Vector3 dVector = particles[p1].position - particles[p2].position;
					Vector3 nVersor = dVector.versor();
					if (dVector.norm() <= particles[p1].radius + particles[p2].radius) {

						// Collision response
						Vector3 cVector = nVersor * 1.0001 * (particles[p1].radius + particles[p2].radius) - dVector;
						particles[p1].position = particles[p1].position + cVector * (particles[p1].mass / (particles[p1].mass + particles[p2].mass));
						particles[p2].position = particles[p2].position - cVector * (particles[p2].mass / (particles[p1].mass + particles[p2].mass));

						Vector3 vrVector = particles[p1].velocity - particles[p2].velocity;
						double nvr = vrVector * nVersor;
						if (abs(nvr) < ZEROTHRESHOLD) {
							Vector3 nVector = nVersor * particles[p1].radius;
							particles[p1].position = (particles[p1].position*particles[p1].mass + particles[p2].position*particles[p2].mass) / (particles[p1].mass + particles[p2].mass);
							particles[p1].velocity = (particles[p1].velocity*particles[p1].mass + particles[p2].velocity*particles[p2].mass) / (particles[p1].mass + particles[p2].mass);
							double newRadius = cbrt(pow(particles[p1].radius, 3) + pow(particles[p2].radius, 3));
							particles[p1].omega = (particles[p1].omega * particles[p1].mass * pow(particles[p1].radius, 2) + particles[p2].omega * particles[p2].mass * pow(particles[p2].radius, 2) + nVector.cross(vrVector) * 2 * particles[p2].mass) / ((particles[p1].mass + particles[p2].mass) * pow(newRadius, 2));
							particles[p1].mass += particles[p2].mass;
							particles[p1].radius = newRadius;
							
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
		double maxOmega = 0;
		for (int p = 0; p < NPARTICLE; p++) {
			if (particles[p].active && (particles[p].omega.norm() > maxOmega)) maxOmega = particles[p].omega.norm();
		}
		debugText = to_string(maxOmega);

		saveFrame(outFile, particles, NPARTICLE);
		frames++;

		if (frameLimit) {
			auto end = chrono::high_resolution_clock::now();
			int deltaTime = chrono::duration_cast<chrono::microseconds>(end - start).count();
			if (deltaTime < targetDt) {
				this_thread::sleep_for(chrono::microseconds(targetDt - deltaTime));
			}
		}
	}
}

void Simulation::integrator(Particle* particles, int NPARTICLE, int FRAMESTEP, int N, double* coefc, double* coefd) {
	Vector3* force = new Vector3[NPARTICLE];
	const double t = 1.0 / (FRAMERATE * FRAMESTEP);

	for (int n = 0; n < N; n++) {
		for (int p1 = 0; p1 < NPARTICLE; p1++) {
			if (!particles[p1].active) continue;
			for (int p2 = p1 + 1; p2 < NPARTICLE; p2++) {
				if (!particles[p2].active) continue;
				Vector3 dVector = particles[p2].position - particles[p1].position;
				double distance = dVector.norm();
				Vector3 curForce = dVector.versor() * particles[p1].mass * particles[p2].mass / pow(distance, 2);
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

void Simulation::saveFrame (ofstream& outFile, Particle* particles, int particleNumber) {
	for (int i = 0; i < particleNumber; i++) {
		if (particles[i].active)
			outFile << particles[i].radius << ":" << particles[i].position.x << "," << particles[i].position.y << "," << particles[i].position.z << ";" << particles[i].angle << "\t";
		else
			outFile << "0" << "\t";
	}
	outFile << "\n";
}

void Simulation::saveProgress(char* ofName, Particle* particles, int particleNumber, int frameStep) {
	ofstream outFile((string(ofName) + string(".dat")).c_str(), ios::out | ios::trunc | ios::binary);
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

void Simulation::printProgress(int currentFrame, int totalFrames) {
	double percentage = ((double)currentFrame / (double)totalFrames) * 100;
	cout << "\r[";
	for (int i = 0; i < 50; i++) {
		if (i == 24) cout << round(percentage) << " %";
		else if (i < round(percentage / 2.0)) cout << "#";
		else cout << " ";
	}
	cout << "]" << flush;
}

void Simulation::terminateSim() {
	outFile.close();
	// saveProgress(outFileName, particles, NPARTICLE, FRAMESTEP); // NOT WORKING RN!

	delete[] particles;
}

void Simulation::stop() {
	
	stopped = true;
}

Particle* Simulation::getParticle() const {
	
	return particles;
}

int Simulation::getNParticle() const {
	
	return NPARTICLE;
}

int* Simulation::getFrameRef() {

	return &frames;
}

void Simulation::setPause(bool pause){

	this->pause = pause;
}

void Simulation::togglePause(){

	pause = !pause;
}