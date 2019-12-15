#include "Simulation.h"

bool stopped = false;
void sigHandler(int sig) {
	cout << flush << "\nRendering stopped, saving progress..." << endl;
	stopped = true;
}

Simulation::Simulation(string outFN, unsigned int duration, unsigned int frameStep, unsigned int particleNumber, bool fl, unsigned long int seed) {
	signal(SIGINT, sigHandler);
	srand(seed);

	// Real simulation or just the viewer
	if ((frameStep != 0) && (particleNumber != 0)){
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
			energyInitialize(particles, NPARTICLE, 0);
		}
	
		outFile.open(outFileName.c_str(), ios::out | ios::binary | (resuming ? ios::app : ios::trunc));
		if (!outFile.is_open()) {
			cerr << "Error opening output file." << endl;
			return;
		}
		if (!resuming) outFile.write((char *)(&NPARTICLE), sizeof(unsigned int));
	}
	else {
		frameLimit = fl;
		outFile.open(outFN, ios::in | ios::binary | ios::ate);
		if (!outFile.is_open()) {
			cerr << "Error opening output file." << endl;
			return;
		}

		unsigned long int fileLength = outFile.tellg();
		outFile.seekg(0, ios::beg);
		outFile.read((char *)(&NPARTICLE), sizeof(unsigned int));
		particles = new Particle[NPARTICLE];
		DURATION = round((float)(fileLength - sizeof(unsigned int)) / (FRAMERATE * NPARTICLE * (sizeof(bool) + sizeof(float) * 8)));
		FRAMESTEP = 0;
	}
}

void Simulation::core() {

	// Second order
	// float cs[] = {0, 1};
	// float cd[] = {0.5, 0.5};

	// Third order
	// float cs[] = {7.0/24, 3.0/4, -1.0/24};
	// float cd[] = {2.0/3, -2.0/3, 1};
	
	// Fourth order
	float cs[] = {0, 1.351207191959657, -1.702414383919315, 1.351207191959657};
	float cd[] = {0.675603595979828, -0.175603595979828, -0.175603595979828, 0.675603595979828};

	bool* active = new bool[NPARTICLE];
	float* radius = new float[NPARTICLE];
	float* Px = new float[NPARTICLE];
	float* Py = new float[NPARTICLE];
	float* Pz = new float[NPARTICLE];
	float* angle = new float[NPARTICLE];
	float* Wx = new float[NPARTICLE];
	float* Wy = new float[NPARTICLE];
	float* Wz = new float[NPARTICLE];

	while ((frames < DURATION * FRAMERATE) && !stopped) {
		auto start = chrono::high_resolution_clock::now();

		if (FRAMESTEP > 0) {
			// Simulation
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
							float nvr = vrVector * nVersor;
							if (abs(nvr) < FUSIONTHRESHOLD) {
								Vector3 nVector = nVersor * particles[p1].radius;
								particles[p1].position = (particles[p1].position*particles[p1].mass + particles[p2].position*particles[p2].mass) / (particles[p1].mass + particles[p2].mass);
								particles[p1].velocity = (particles[p1].velocity*particles[p1].mass + particles[p2].velocity*particles[p2].mass) / (particles[p1].mass + particles[p2].mass);
								float newRadius = cbrt(pow(particles[p1].radius, 3) + pow(particles[p2].radius, 3));
								particles[p1].omega = (particles[p1].omega * particles[p1].mass * pow(particles[p1].radius, 2) + particles[p2].omega * particles[p2].mass * pow(particles[p2].radius, 2) + nVector.cross(vrVector) * 5 * particles[p2].mass) / ((particles[p1].mass + particles[p2].mass) * pow(newRadius, 2));
								particles[p1].mass += particles[p2].mass;
								particles[p1].radius = newRadius;
								
								particles[p2].active = false;
							}
							else {
								float Jr = ((EFACTOR + 1) / (1 / particles[p1].mass + 1 / particles[p2].mass)) * nvr;
								particles[p1].velocity = particles[p1].velocity - nVersor * (Jr / particles[p1].mass);
								particles[p2].velocity = particles[p2].velocity + nVersor * (Jr / particles[p2].mass);
							}
						}
					}
				}
			}
			saveFrame(outFile, particles, NPARTICLE);
		}
		else {
			// Just viewer
			outFile.read((char *)active, NPARTICLE * sizeof(bool));
			outFile.read((char *)radius, NPARTICLE * sizeof(float));
			outFile.read((char *)Px, NPARTICLE * sizeof(float));
			outFile.read((char *)Py, NPARTICLE * sizeof(float));
			outFile.read((char *)Pz, NPARTICLE * sizeof(float));
			outFile.read((char *)angle, NPARTICLE * sizeof(float));
			outFile.read((char *)Wx, NPARTICLE * sizeof(float));
			outFile.read((char *)Wy, NPARTICLE * sizeof(float));
			outFile.read((char *)Wz, NPARTICLE * sizeof(float));

			for (unsigned int p = 0; p < NPARTICLE; p++) {
				particles[p].active = active[p];
				particles[p].radius = radius[p];
				particles[p].position = Vector3(Px[p], Py[p], Pz[p]);
				particles[p].angle = angle[p];				
				particles[p].omega = Vector3(Wx[p], Wy[p], Wz[p]);
			}
		}

		frames++;
		auto end = chrono::high_resolution_clock::now();
		int deltaTime = chrono::duration_cast<chrono::microseconds>(end - start).count();
		// debugText = to_string(1000000 / deltaTime) + " fps";
		if (frameLimit) {
			if (deltaTime < targetDt) {
				this_thread::sleep_for(chrono::microseconds(targetDt - deltaTime));
			}
		}
	}
}

void Simulation::randomInitialize (Particle* particles, int NPARTICLE) {
	float R = cbrt( (3*NPARTICLE) / (4*PI*INITDENSITY) );
	for (int p = 0; p < NPARTICLE; p++) {
		Vector3 rpVector (random(-R, R), random(-R, R), random(-R, R));
		while (rpVector.norm() > R) {
			rpVector = Vector3(random(-R, R), random(-R, R), random(-R, R));
		}
		particles[p].position = rpVector;
		particles[p].velocity = rpVector.versor() * random(0, 5);
		particles[p].omega = Vector3(random(-1, 1), random(-1, 1), random(-1, 1));
		particles[p].mass = random(0.5, 2);
		particles[p].radius = random(0.1, 0.2);
	}
}

void Simulation::energyInitialize (Particle* particles, int NPARTICLE, float E0) {
	float R = cbrt( (3*NPARTICLE) / (4*PI*INITDENSITY) );
	for (int p = 0; p < NPARTICLE; p++) {
		Vector3 rpVector (random(-R, R), random(-R, R), random(-R, R));
		while (rpVector.norm() > R) {
			rpVector = Vector3(random(-R, R), random(-R, R), random(-R, R));
		}
		particles[p].position = rpVector;
		particles[p].mass = random(0.5, 2);
		particles[p].radius = random(0.1, 0.2);
	}

	do {
		float targetv2 = 2 * (E0 - getPotential(particles, NPARTICLE)) / (1.25 * NPARTICLE);
		for (int p = 0; p < NPARTICLE; p++) {
			particles[p].velocity = particles[p].position.versor() * sqrt(random(0, 2*targetv2));
		}
	} while (abs(getEnergy(particles, NPARTICLE) - E0) > INITTHRESHOLD);
}

float Simulation::getKinetic(Particle* particles, int NPARTICLE) {
	float K = 0;
	for (int p = 0; p < NPARTICLE; p++){
		K += 0.5 * particles[p].mass * pow(particles[p].velocity.norm(), 2) + 0.2 * particles[p].mass * pow(particles[p].radius * particles[p].omega.norm(), 2);
	}
	return K;
}

float Simulation::getPotential(Particle* particles, int NPARTICLE) {
	float U = 0;
	for (int p1 = 0; p1 < NPARTICLE; p1++){
		for (int p2 = 0; p2 < NPARTICLE; p2++){
			if (p1 == p2) continue;
			U -= (particles[p1].mass * particles[p2].mass) / (particles[p1].position - particles[p2].position).norm();
		}
	}
	return U;
}

float Simulation::getEnergy(Particle* particles, int NPARTICLE) {

	return getKinetic(particles, NPARTICLE) + getPotential(particles, NPARTICLE);
}

void Simulation::integrator(Particle* particles, int NPARTICLE, int FRAMESTEP, int N, float* coefc, float* coefd) {
	Vector3* force = new Vector3[NPARTICLE];
	const float t = 1.0 / (FRAMERATE * FRAMESTEP);

	for (int n = 0; n < N; n++) {
		for (int p1 = 0; p1 < NPARTICLE; p1++) {
			if (!particles[p1].active) continue;
			for (int p2 = p1 + 1; p2 < NPARTICLE; p2++) {
				if (!particles[p2].active) continue;
				Vector3 dVector = particles[p2].position - particles[p1].position;
				float distance = dVector.norm();
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

void Simulation::saveFrame (fstream& outFile, Particle* particles, int particleNumber) {
	for (int i = 0; i < particleNumber; i++) {
		outFile.write((char *)(&(particles[i].active)), sizeof(bool));
	}	
	for (int i = 0; i < particleNumber; i++) {
		outFile.write((char *)(&(particles[i].radius)), sizeof(float));
	}	
	for (int i = 0; i < particleNumber; i++) {
		outFile.write((char *)(&(particles[i].position.x)), sizeof(float));
	}	
	for (int i = 0; i < particleNumber; i++) {
		outFile.write((char *)(&(particles[i].position.y)), sizeof(float));
	}	
	for (int i = 0; i < particleNumber; i++) {
		outFile.write((char *)(&(particles[i].position.z)), sizeof(float));
	}	
	for (int i = 0; i < particleNumber; i++) {
		outFile.write((char *)(&(particles[i].angle)), sizeof(float));
	}	
	for (int i = 0; i < particleNumber; i++) {
		outFile.write((char *)(&(particles[i].omega.x)), sizeof(float));
	}	
	for (int i = 0; i < particleNumber; i++) {
		outFile.write((char *)(&(particles[i].omega.y)), sizeof(float));
	}	
	for (int i = 0; i < particleNumber; i++) {
		outFile.write((char *)(&(particles[i].omega.z)), sizeof(float));
	}
}

void Simulation::saveProgress(char* ofName, Particle* particles, int particleNumber, int frameStep) {
	fstream outFile((string(ofName) + string(".dat")).c_str(), ios::out | ios::trunc | ios::binary);
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
	float percentage = ((float)currentFrame / (float)totalFrames) * 100;
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