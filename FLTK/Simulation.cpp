#include "Simulation.h"

bool stopped = false;
void sigHandler(int sig) {
	cout << flush << "\nRendering stopped, saving progress..." << endl;
	stopped = true;
}

Simulation::Simulation(int argc, char** argv) {
	signal(SIGINT, sigHandler);
	char* outFileName;

	// Optional terminal paramenters IN THIS ORDER!
	DURATION = 60, FRAMESTEP = 10, NPARTICLE = 3;

	if (argc > 1) outFileName = argv[argc - 1];
	else {
		cerr << "You must specify an output file.\nUsage: ./core [DURATION] [FRAMESTEP] [NPARTICLE] \"OutputFile.txt\"\nor\n./core \"ProgressData.dat\"" << endl;
		exit(1);
	}
	bool resuming = string(outFileName).substr(string(outFileName).length() - 4, 4).compare(string(".dat")) == 0;
	if (!resuming) {
		if (argc > 2) NPARTICLE = stoi(argv[argc - 2]);
		if (argc > 3) FRAMESTEP = stoi(argv[argc - 3]);
		if (argc > 4) DURATION = stoi(argv[argc - 4]);
	}
	else {
		if (argc > 2) DURATION = stoi(argv[argc - 2]);
	}

	particles = new Particle[NPARTICLE];
	if (resuming) {
		ifstream inFile(outFileName, ios::in | ios::binary);
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
		strcpy(outFileName, string(outFileName).substr(0, string(outFileName).length() - 4).c_str());
	}
	else {
		// for (int p = 0; p < NPARTICLE; p++) {
		// 	Vector2 rpVector(random(-10, 10), random(-10, 10));
		// 	particles[p].position = rpVector;
		// 	particles[p].velocity = rpVector.versor() * random(0, 5);
		// 	// particles[p].radius = 0.5;
		// 	// particles[p].mass = 10;
		// 	// particles[p].velocity = Vector2(-rpVector.y, rpVector.x).versor() * random(0, 10);
		// 	// particles[p].velocity = Vector2(random(-10, 10), random(-10, 10));
		// 	// particles[p].position = Vector2(random(-80, 80), random(-50, 50));
		// }

		particles[0].position = Vector2(0, 0);
		particles[0].velocity = Vector2(0, 0);
		particles[1].position = Vector2(-0.05, -0.05);
		particles[1].velocity = Vector2(0, 0);
		particles[2].position = Vector2(0.05, -0.05);
		particles[2].velocity = Vector2(0, 1);
	}

	outFile.open(outFileName, ios::out | (resuming ? ios::app : ios::trunc));
	if (!outFile.is_open()) {
		cerr << "Error opening output file." << endl;
		return;
	}
	outFile << particles[0].radius << "\n";	
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
			
			//It checks if the simulation is in pause, wait for 0.1 second to recheck the variable
			while(pause){ this_thread::sleep_for(chrono::milliseconds(100)); }

			// Collision detection (discrete)
			for (int p1 = 0; p1 < NPARTICLE; p1++) {
				for (int p2 = p1 + 1; p2 < NPARTICLE; p2++) {
					Vector2 dVector = particles[p1].position - particles[p2].position;
					Vector2 nVersor = dVector.versor();
					if (dVector.norm() <= particles[p1].radius + particles[p2].radius) {

						// Collision response
						Vector2 cVector = nVersor * 1.0001 * (particles[p1].radius + particles[p2].radius) - dVector;
						particles[p1].position = particles[p1].position + cVector / 2;
						particles[p2].position = particles[p2].position - cVector / 2;

						Vector2 vrVector = particles[p1].velocity - particles[p2].velocity;
						double Jr = ((EFACTOR + 1) / (1 / particles[p1].mass + 1 / particles[p2].mass)) * (vrVector * nVersor);
						particles[p1].velocity = particles[p1].velocity - nVersor * (Jr / particles[p1].mass);
						particles[p2].velocity = particles[p2].velocity + nVersor * (Jr / particles[p2].mass);
					}
				}
			}
		}
		saveFrame(outFile, particles, NPARTICLE);
		frames++;

		auto end = chrono::high_resolution_clock::now();
		int deltaTime = chrono::duration_cast<chrono::microseconds>(end - start).count();
		if (deltaTime < targetDt) {
			this_thread::sleep_for(chrono::microseconds(targetDt - deltaTime));
		}
		// if (!stopped) printProgress(frames, DURATION * FRAMERATE);
	}
}

void Simulation::integrator(Particle* particles, int NPARTICLE, int FRAMESTEP, int N, double* coefc, double* coefd) {
	Vector2* force = new Vector2[NPARTICLE];
	const double t = 1.0 / (FRAMERATE * FRAMESTEP);

	for (int n = 0; n < N; n++) {
		for (int p1 = 0; p1 < NPARTICLE; p1++) {
			for (int p2 = p1 + 1; p2 < NPARTICLE; p2++) {
				Vector2 dVector = particles[p2].position - particles[p1].position;
				double distance = dVector.norm();
				Vector2 curForce = dVector.versor() * particles[p1].mass * particles[p2].mass / pow(distance, 2);
				force[p1] = force[p1] + curForce;
				force[p2] = force[p2] - curForce;
			}
		}
		for (int p = 0; p < NPARTICLE; ++p) {
			particles[p].velocityStep(coefc[n] * t, force[p]);
			particles[p].positionStep(coefd[n] * t, force[p]);
			force[p] = particles[p].position * -KFACTOR;
		}
	}

	delete[] force;
}

void Simulation::saveFrame(ofstream& outFile, Particle* particles, int particleNumber) {
	for (int i = 0; i < particleNumber; i++) {
		outFile << particles[i].position.x << "," << particles[i].position.y << "\t";
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

double Simulation::random(double min, double max) {
	// Useless comment so I can fold this function in Sublime Text :)
	return ((double)rand() / RAND_MAX) * (max - min) + min;
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
	// cout << "\n";
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

int* Simulation::getFrameRef()
{
	return &frames;
}

void Simulation::setPause(bool pause){

	this->pause = pause;
}

void Simulation::togglePause(){

	pause = !pause;
}