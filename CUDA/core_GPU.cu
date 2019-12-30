#include <iostream>
#include <fstream>
#include <string>
#include <signal.h>
#define FRAMERATE 30
#define EFACTOR 0.2
#define KFACTOR 0 // 0.005
#define SCHEMEORDER 4
#define FUSIONTHRESHOLD 0.2
#define INITTHRESHOLD 0.05
#define INITDENSITY 0.5
#define PI 3.14159265359

#define DATAPERIOD 30 // 0 to disable distributions output
#define BASEDATADIR "Distributions/"
using namespace std;

__host__ __device__ float norm (float x, float y, float z) {

	return sqrtf(x*x + y*y + z*z);
}

float getKinetic (bool* active, float* mass, float* Vx, float* Vy, float* Vz, float* radius, float* Wx, float* Wy, float* Wz, unsigned int NPARTICLE) {
	float K = 0;
	for (int p = 0; p < NPARTICLE; p++) {
		if (active[p]) K += 0.5 * mass[p] * (Vx[p]*Vx[p] + Vy[p]*Vy[p] + Vz[p]*Vz[p]) + 0.2 * mass[p] * radius[p]*radius[p] * (Wx[p]*Wx[p] + Wy[p]*Wy[p] + Wz[p]*Wz[p]);
	}
	return K;
}

float getPotential (bool* active, float* mass, float* Px, float* Py, float* Pz, unsigned int NPARTICLE) {
	float U = 0;
	for (int p1 = 0; p1 < NPARTICLE; p1++) {
		if (!active[p1]) continue;
		for (int p2 = p1 + 1; p2 < NPARTICLE; p2++) {
			if (active[p2]) U -= 2 * mass[p1] * mass[p2] / norm(Px[p1]-Px[p2], Py[p1]-Py[p2], Pz[p1]-Pz[p2]);
		}
	}
	return U;
}

float getEnergy (bool* active, float* mass, float* Px, float* Py, float* Pz, float* Vx, float* Vy, float* Vz, float* radius, float* Wx, float* Wy, float* Wz, unsigned int NPARTICLE) {

	return getKinetic(active, mass, Vx, Vy, Vz, radius, Wx, Wy, Wz, NPARTICLE) + getPotential(active, mass, Px, Py, Pz, NPARTICLE);
}

float getSingleEnergy (bool* active, float* mass, float* Px, float* Py, float* Pz, float* Vx, float* Vy, float* Vz, float* radius, float* Wx, float* Wy, float* Wz, unsigned int NPARTICLE, int p) {
	float E = 0;
	E += 0.5 * mass[p] * (Vx[p]*Vx[p] + Vy[p]*Vy[p] + Vz[p]*Vz[p]) + 0.2 * mass[p] * powf(radius[p], 2) * (Wx[p]*Wx[p] + Wy[p]*Wy[p] + Wz[p]*Wz[p]);
	for (int p2 = 0; p2 < NPARTICLE; p2++) {
		if ((p == p2) || !active[p2]) continue;
		E -= mass[p] * mass[p2] / norm(Px[p]-Px[p2], Py[p]-Py[p2], Pz[p]-Pz[p2]);
	}
	return E;
}

void saveFrame (ofstream& outFile, float* radius, float* Px, float* Py, float* Pz, float* angle, float* Wx, float* Wy, float* Wz, int particleNumber, bool* active) {
	outFile.write((char *)active, particleNumber * sizeof(bool));
	outFile.write((char *)radius, particleNumber * sizeof(float));
	outFile.write((char *)Px, particleNumber * sizeof(float));
	outFile.write((char *)Py, particleNumber * sizeof(float));
	outFile.write((char *)Pz, particleNumber * sizeof(float));
	outFile.write((char *)angle, particleNumber * sizeof(float));
	outFile.write((char *)Wx, particleNumber * sizeof(float));
	outFile.write((char *)Wy, particleNumber * sizeof(float));
	outFile.write((char *)Wz, particleNumber * sizeof(float));
}

void saveSize (ofstream& outFile, float t, float* Px, float* Py, float* Pz, float* mass, int particleNumber, bool* active) {
	float curSize = 0, totalWeight = 0;
	for (int p1 = 0; p1 < particleNumber; p1++) {
		if (!active[p1]) continue;
		for (int p2 = p1+1; p2 < particleNumber; p2++) {
			if (!active[p2]) continue;
			curSize += norm(Px[p2]-Px[p1], Py[p2]-Py[p1], Pz[p2]-Pz[p1]) * mass[p1]*mass[p2];
			totalWeight += mass[p1]*mass[p2];
		}
	}
	curSize /= totalWeight;
	outFile << t << '\t' << curSize << '\n';
}

float random (float min, float max) {

	return ((float)rand() / RAND_MAX) * (max-min) + min;
}

void printProgress (unsigned int currentFrame, unsigned int totalFrames) {
	float percentage = ((float)currentFrame / (float)totalFrames) * 100;
	cout << "\r[";
	for (int i = 0; i < 50; i++) {
		if (i == 24) cout << round(percentage) << " %";
		else if (i < round(percentage/2.0)) cout << "#";
		else cout << " ";
	}
	cout << "]" << flush;
}

void saveProgress (string ofName, unsigned int particleNumber, unsigned int frameStep, bool* active, float* radius, float* mass, float* Px, float* Py, float* Pz, float* angle, float* Wx, float* Wy, float* Wz, float* Vx, float* Vy, float* Vz) {
	ofstream outFile ((ofName + string(".dat")).c_str(), ios::out | ios::trunc | ios::binary);
	if (!outFile.is_open()) {
		cerr << "Error opening output file." << endl;
		return;
	}

	outFile.write((char *)(&particleNumber), sizeof(unsigned int));
	outFile.write((char *)(&frameStep), sizeof(unsigned int));

	outFile.write((char *)active, particleNumber * sizeof(bool));
	outFile.write((char *)radius, particleNumber * sizeof(float));
	outFile.write((char *)mass, particleNumber * sizeof(float));
	outFile.write((char *)Px, particleNumber * sizeof(float));
	outFile.write((char *)Py, particleNumber * sizeof(float));
	outFile.write((char *)Pz, particleNumber * sizeof(float));
	outFile.write((char *)Vx, particleNumber * sizeof(float));
	outFile.write((char *)Vy, particleNumber * sizeof(float));
	outFile.write((char *)Vz, particleNumber * sizeof(float));
	outFile.write((char *)angle, particleNumber * sizeof(float));
	outFile.write((char *)Wx, particleNumber * sizeof(float));
	outFile.write((char *)Wy, particleNumber * sizeof(float));
	outFile.write((char *)Wz, particleNumber * sizeof(float));

	outFile.close();
}

bool stopped = false;
void sigHandler (int sig) {
	cout << flush << "\nRendering stopped, saving progress..." << endl;
	stopped = true;
}

void randomInitialize (float* mass, float* radius, float* Px, float* Py, float* Pz, float* Vx, float* Vy, float* Vz, float* Wx, float* Wy, float* Wz, unsigned int NPARTICLE) {
	float R = cbrt( (3*NPARTICLE) / (4*PI*INITDENSITY) );
	for (int p = 0; p < NPARTICLE; p++) {

		// Positions
		float rx = random(-R, R), ry = random(-R, R), rz = random(-R, R), rnorm = norm(rx, ry, rz);
		while (rnorm > R) {
			rx = random(-R, R);
			ry = random(-R, R);
			rz = random(-R, R);
			rnorm = norm(rx, ry, rz);
		}
		Px[p] = rx;
		Py[p] = ry;
		Pz[p] = rz;

		// Velocities
		float rvn = random(0, 5);
		Vx[p] = (rx/rnorm) * rvn;
		Vy[p] = (ry/rnorm) * rvn;
		Vz[p] = (rz/rnorm) * rvn;

		// Omegas
		Wx[p] = random(-1, 1);
		Wy[p] = random(-1, 1);
		Wz[p] = random(-1, 1);

		// Masses and radii
		mass[p] = random(0.5, 2);
		radius[p] = random(0.1, 0.2);
	}
}

float energyInitialize (float* mass, float* radius, float* Px, float* Py, float* Pz, float* Vx, float* Vy, float* Vz, float* Wx, float* Wy, float* Wz, bool* active, unsigned int NPARTICLE, float E0) {
	cout << "Initializing system with a target energy..." << endl;
	float R = cbrt( (3*NPARTICLE) / (4*PI*INITDENSITY) );
	float targetv2 = -1;
	while ((targetv2 < 0) && !stopped) {
		int tried = 0;
		while ((targetv2 < 0) && !stopped && tried < 1000) {
			for (int p = 0; p < NPARTICLE; p++) {

				// Positions
				float rx = random(-R, R), ry = random(-R, R), rz = random(-R, R), rnorm = norm(rx, ry, rz);
				while (rnorm > R) {
					rx = random(-R, R);
					ry = random(-R, R);
					rz = random(-R, R);
					rnorm = norm(rx, ry, rz);
				}
				Px[p] = rx;
				Py[p] = ry;
				Pz[p] = rz;

				// Masses and radii
				mass[p] = random(0.5, 2);
				radius[p] = random(0.1, 0.2);
			}
			targetv2 = 2 * (E0 - getPotential(active, mass, Px, Py, Pz, NPARTICLE)) / (1.25 * NPARTICLE);
			tried ++;
		}
		if (tried >= 1000) {
			cout << "Failed to find an appropriate kinetic energy, making universe 10% smaller: R = " << R << "..." << endl;
			R *= 0.9;
		}
	}

	// Velocities
	float curEnergy = 0;
	do {
		for (int p = 0; p < NPARTICLE; p++) {
			float rnorm = norm(Px[p], Py[p], Pz[p]), rvn = sqrtf(random(0.8*targetv2, 1.2*targetv2));
			Vx[p] = (Px[p]/rnorm) * rvn;
			Vy[p] = (Py[p]/rnorm) * rvn;
			Vz[p] = (Pz[p]/rnorm) * rvn;
		}
		curEnergy = getEnergy(active, mass, Px, Py, Pz, Vx, Vy, Vz, radius, Wx, Wy, Wz, NPARTICLE);
	} while ((abs(curEnergy - E0) > INITTHRESHOLD*abs(E0))  && !stopped);
	cout << "System initialized with E = " << curEnergy << endl;
	return curEnergy;
}

__global__ void computeForces (float* Fx, float* Fy, float* Fz, float* Px, float* Py, float* Pz, float* mass, int NPARTICLE, bool* active) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < NPARTICLE*NPARTICLE) {
		int p1 = p / NPARTICLE, p2 = p % NPARTICLE;
		if ((p2 < NPARTICLE) && (p2 > p1) && active[p1] && active[p2]) {
			float Dx = Px[p2]-Px[p1], Dy = Py[p2]-Py[p1], Dz = Pz[p2]-Pz[p1];
			float distance2 = Dx*Dx + Dy*Dy + Dz*Dz, distance = sqrtf(distance2);

			float curFx = (Dx/distance) * mass[p1] * mass[p2] / distance2;
			float curFy = (Dy/distance) * mass[p1] * mass[p2] / distance2;
			float curFz = (Dz/distance) * mass[p1] * mass[p2] / distance2;
			atomicAdd(Fx+p1, curFx);
			atomicAdd(Fy+p1, curFy);
			atomicAdd(Fz+p1, curFz);
			atomicAdd(Fx+p2, -curFx);
			atomicAdd(Fy+p2, -curFy);
			atomicAdd(Fz+p2, -curFz);
		}
	}
}

__global__ void moveSystem (float* Fx, float* Fy, float* Fz, float* Px, float* Py, float* Pz, float* Vx, float* Vy, float* Vz, float* angle, float* Wx, float* Wy, float* Wz, float* mass, float dtv, float dtp, float dta, int NPARTICLE, bool* active) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < NPARTICLE && active[p]) {
		Vx[p] += (Fx[p]/mass[p]) * dtv;
		Vy[p] += (Fy[p]/mass[p]) * dtv;
		Vz[p] += (Fz[p]/mass[p]) * dtv;
		Px[p] += Vx[p] * dtp;
		Py[p] += Vy[p] * dtp;
		Pz[p] += Vz[p] * dtp;
		angle[p] = fmodf(angle[p] + norm(Wx[p], Wy[p], Wz[p]) * dta, 2*PI);

		Fx[p] = -KFACTOR * Px[p];
		Fy[p] = -KFACTOR * Py[p];
		Fz[p] = -KFACTOR * Pz[p];
	}
}

__global__ void computeCollisions (float* Px, float* Py, float* Pz, float* Vx, float* Vy, float* Vz, float* Wx, float* Wy, float* Wz, float* Sx, float* Sy, float* Sz, float* Jx, float* Jy, float* Jz,float* mass, float* radius, int NPARTICLE, bool* active) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < NPARTICLE*NPARTICLE) {
		int p1 = p / NPARTICLE, p2 = p % NPARTICLE;
		if ((p2 < NPARTICLE) && (p2 > p1) && active[p1] && active[p2]) {
			float Dx = Px[p1] - Px[p2], Dy = Py[p1] - Py[p2], Dz = Pz[p1] - Pz[p2];
			float distance = norm(Dx, Dy, Dz);
			float Nx = Dx/distance, Ny = Dy/distance, Nz = Dz/distance;
			if (distance <= radius[p1] + radius[p2]) {
				float Cx = Nx * radius[p1], Cy = Ny * radius[p1], Cz = Nz * radius[p1];
				float Vrx = Vx[p1] - Vx[p2], Vry = Vy[p1] - Vy[p2], Vrz = Vz[p1] - Vz[p2];
				float nvr = Vrx*Nx + Vry*Ny + Vrz*Nz;
				if (abs(nvr) > FUSIONTHRESHOLD) {

					// Collision
					float cSx = Nx * 1.0001*(radius[p1] + radius[p2]) - Dx;
					float cSy = Ny * 1.0001*(radius[p1] + radius[p2]) - Dy;
					float cSz = Nz * 1.0001*(radius[p1] + radius[p2]) - Dz;
					atomicAdd( Sx+p1, cSx * mass[p1] / (mass[p1] + mass[p2]) );
					atomicAdd( Sy+p1, cSy * mass[p1] / (mass[p1] + mass[p2]) );
					atomicAdd( Sz+p1, cSz * mass[p1] / (mass[p1] + mass[p2]) );
					atomicAdd( Sx+p2, -cSx * mass[p2] / (mass[p1] + mass[p2]) );
					atomicAdd( Sy+p2, -cSy * mass[p2] / (mass[p1] + mass[p2]) );
					atomicAdd( Sz+p2, -cSz * mass[p2] / (mass[p1] + mass[p2]) );

					float cJr = ((EFACTOR+1)/(1/mass[p1] + 1/mass[p2])) * nvr;
					atomicAdd( Jx+p1, -Nx * (cJr/mass[p1]) );
					atomicAdd( Jy+p1, -Ny * (cJr/mass[p1]) );
					atomicAdd( Jz+p1, -Nz * (cJr/mass[p1]) );
					atomicAdd( Jx+p2, Nx * (cJr/mass[p2]) );
					atomicAdd( Jy+p2, Ny * (cJr/mass[p2]) );
					atomicAdd( Jz+p1, -Nz * (cJr/mass[p1]) );
				}
				else {
					// Fusion
					Px[p1] = (Px[p1] * mass[p1] + Px[p2] * mass[p2]) / (mass[p1] + mass[p2]);
					Py[p1] = (Py[p1] * mass[p1] + Py[p2] * mass[p2]) / (mass[p1] + mass[p2]);
					Pz[p1] = (Pz[p1] * mass[p1] + Pz[p2] * mass[p2]) / (mass[p1] + mass[p2]);

					Vx[p1] = (Vx[p1] * mass[p1] + Vx[p2] * mass[p2]) / (mass[p1] + mass[p2]);
					Vy[p1] = (Vy[p1] * mass[p1] + Vy[p2] * mass[p2]) / (mass[p1] + mass[p2]);
					Vz[p1] = (Vz[p1] * mass[p1] + Vz[p2] * mass[p2]) / (mass[p1] + mass[p2]);

					float newRadius = cbrtf(pow(radius[p1], 3) + pow(radius[p2], 3));
					Wx[p1] = (mass[p1] * pow(radius[p1], 2) * Wx[p1] + mass[p2] * pow(radius[p2], 2) * Wx[p2] + 5 * mass[p2] * (Cy*Vrz - Cz*Vry)) / ((mass[p1] + mass[p2]) * pow(newRadius, 2));
					Wy[p1] = (mass[p1] * pow(radius[p1], 2) * Wy[p1] + mass[p2] * pow(radius[p2], 2) * Wy[p2] + 5 * mass[p2] * (Cz*Vrx - Cx*Vrz)) / ((mass[p1] + mass[p2]) * pow(newRadius, 2));
					Wz[p1] = (mass[p1] * pow(radius[p1], 2) * Wz[p1] + mass[p2] * pow(radius[p2], 2) * Wz[p2] + 5 * mass[p2] * (Cx*Vry - Cy*Vrx)) / ((mass[p1] + mass[p2]) * pow(newRadius, 2));
					mass[p1] += mass[p2];
					radius[p1] = newRadius;

					active[p2] = false;
				}
			}
		}
	}
}

__global__ void collisionResponse (float* Px, float* Py, float* Pz, float* Vx, float* Vy, float* Vz, float* Sx, float* Sy, float* Sz, float* Jx, float* Jy, float* Jz, int NPARTICLE, bool* active) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < NPARTICLE && active[p]) {
		Px[p] += Sx[p];
		Py[p] += Sy[p];
		Pz[p] += Sz[p];

		Vx[p] += Jx[p];
		Vy[p] += Jy[p];
		Vz[p] += Jz[p];

		Sx[p] = Sy[p] = Sz[p] = Jx[p] = Jy[p] = Jz[p] = 0;
	}
}

int main(int argc, char** argv) {
	signal(SIGINT, sigHandler);
	srand(time(0) * rand());

	string outFileName;
	unsigned int DURATION = 60, FRAMESTEP = 10, NPARTICLE = 500;
	float ENERGY0 = 0, realE0 = 0;
	if (argc > 1) outFileName = string(argv[argc-1]);
	else {
		cerr << "You must specify an output or save file.\nUsage: ./core [DURATION] [FRAMESTEP] [NPARTICLE] \"OutputFile.txt\"\nor\n./core \"ProgressData.dat\"" << endl;
		return 1;
	}
	bool resuming = outFileName.substr(outFileName.length()-4, 4).compare(string(".dat")) == 0;
	if (!resuming) {
		if (argc > 2) ENERGY0 = stof(argv[argc-2]);
		if (argc > 3) NPARTICLE = stoi(argv[argc-3]);
		if (argc > 4) FRAMESTEP = stoi(argv[argc-4]);
		if (argc > 5) DURATION = stoi(argv[argc-5]);
	}
	else {
		if (argc > 2) DURATION = stoi(argv[argc-2]);
	}
	if (DURATION == 0) DURATION = UINT_MAX;
	ENERGY0 *= NPARTICLE;
	
	float *mass, *radius, *Px, *Py, *Pz, *Vx, *Vy, *Vz, *angle, *Wx, *Wy, *Wz, *Fx, *Fy, *Fz, *Sx, *Sy, *Sz, *Jx, *Jy, *Jz;
	bool *active;
	cudaMallocManaged(&mass, NPARTICLE*sizeof(float));
	cudaMallocManaged(&radius, NPARTICLE*sizeof(float));
	cudaMallocManaged(&angle, NPARTICLE*sizeof(float));

	cudaMallocManaged(&Px, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Py, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Pz, NPARTICLE*sizeof(float));

	cudaMallocManaged(&Vx, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Vy, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Vz, NPARTICLE*sizeof(float));

	cudaMallocManaged(&Wx, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Wy, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Wz, NPARTICLE*sizeof(float));

	cudaMallocManaged(&Fx, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Fy, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Fz, NPARTICLE*sizeof(float));

	cudaMallocManaged(&Sx, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Sy, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Sz, NPARTICLE*sizeof(float));

	cudaMallocManaged(&Jx, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Jy, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Jz, NPARTICLE*sizeof(float));

	cudaMallocManaged(&active, NPARTICLE*sizeof(bool));

	// Initialization objects
	for (int p = 0; p < NPARTICLE; p++) {
		active[p] = true;
		angle[p] = 0;

		Fx[p] = -KFACTOR * Px[p];
		Fy[p] = -KFACTOR * Py[p];
		Fz[p] = -KFACTOR * Pz[p];

		Sx[p] = Sy[p] = Sz[p] = Jx[p] = Jy[p] = Jz[p] = 0;
	}
	if (resuming) {
		cout << "Save file specified, restoring progress..." << endl;
		ifstream inFile (outFileName.c_str(), ios::in | ios::binary);
		if (!inFile.is_open()) {
			cerr << "Error opening input file." << endl;
			return 2;
		}

		inFile.read((char *)(&NPARTICLE), sizeof(unsigned int));
		inFile.read((char *)(&FRAMESTEP), sizeof(unsigned int));

		inFile.read((char *)active, NPARTICLE * sizeof(bool));
		inFile.read((char *)radius, NPARTICLE * sizeof(float));
		inFile.read((char *)mass, NPARTICLE * sizeof(float));
		inFile.read((char *)Px, NPARTICLE * sizeof(float));
		inFile.read((char *)Py, NPARTICLE * sizeof(float));
		inFile.read((char *)Pz, NPARTICLE * sizeof(float));
		inFile.read((char *)Vx, NPARTICLE * sizeof(float));
		inFile.read((char *)Vy, NPARTICLE * sizeof(float));
		inFile.read((char *)Vz, NPARTICLE * sizeof(float));
		inFile.read((char *)angle, NPARTICLE * sizeof(float));
		inFile.read((char *)Wx, NPARTICLE * sizeof(float));
		inFile.read((char *)Wy, NPARTICLE * sizeof(float));
		inFile.read((char *)Wz, NPARTICLE * sizeof(float));

		inFile.close();
		outFileName = outFileName.substr(0, outFileName.length()-4);
	}
	else {
		// randomInitialize(mass, radius, Px, Py, Pz, Vx, Vy, Vz, Wx, Wy, Wz, NPARTICLE);
		realE0 = energyInitialize(mass, radius, Px, Py, Pz, Vx, Vy, Vz, Wx, Wy, Wz, active, NPARTICLE, ENERGY0);
	}

	// Output file initialization
	ofstream outFile (outFileName.c_str(), ios::out | ios::binary | (resuming ? ios::app : ios::trunc));
	if (!outFile.is_open()) {
		cerr << "Error opening output file." << endl;
		return 2;
	}
	if (!resuming) outFile.write((char *)(&NPARTICLE), sizeof(unsigned int));

	const string DATADIR = string(BASEDATADIR) + string("E") + to_string(realE0) + string("/");
	system( (string("mkdir -p \"") + DATADIR + string("\"")).c_str() );
	string dataFileName = string(DATADIR) + string("size.dat");
	ofstream sizeFile (dataFileName.c_str(), ios::out | ios::trunc);

	float cs[] = {0, 0, 0, 0};
	float cd[] = {0, 0, 0, 0};
	if (SCHEMEORDER == 2) {
		cs[0] = 0; cs[1] = 1;
		cd[0] = 0.5; cd[1] = 0.5;
	}

	if (SCHEMEORDER == 3) {
		cs[0] = 7.0/24; cs[1] = 3.0/4; cs[2] = -1.0/24;
		cd[0] = 2.0/3; cd[1] = -2.0/3; cd[2] = 1;
	}

	if (SCHEMEORDER == 4) {
		cs[0] = 0; cs[1] = 1.351207191959657; cs[2] = -1.702414383919315; cs[3] = 1.351207191959657;
		cd[0] = 0.675603595979828; cd[1] = -0.175603595979828; cd[2] = -0.175603595979828; cd[3] = 0.675603595979828;
	}

	unsigned int frames = 0;
	const float dt = 1.0/(FRAMERATE*FRAMESTEP);
	while ((frames < DURATION*FRAMERATE) && (!stopped)) {
		for (int i = 0; i < FRAMESTEP; i ++) {

			for (int s = 0; s < SCHEMEORDER; s ++) {
				computeForces <<<(NPARTICLE*NPARTICLE+255)/256, 256>>> (Fx, Fy, Fz, Px, Py, Pz, mass, NPARTICLE, active);
				cudaDeviceSynchronize();

				moveSystem <<<(NPARTICLE+255)/256, 256>>> (Fx, Fy, Fz, Px, Py, Pz, Vx, Vy, Vz, angle, Wx, Wy, Wz, mass, cs[s]*dt, cd[s]*dt, dt/SCHEMEORDER, NPARTICLE, active);
				cudaDeviceSynchronize();
			}

			computeCollisions <<<(NPARTICLE*NPARTICLE+255)/256, 256>>> (Px, Py, Pz, Vx, Vy, Vz, Wx, Wy, Wz, Sx, Sy, Sz, Jx, Jy, Jz, mass, radius, NPARTICLE, active);
			cudaDeviceSynchronize();

			collisionResponse <<<(NPARTICLE+255)/256, 256>>> (Px, Py, Pz, Vx, Vy, Vz, Sx, Sy, Sz, Jx, Jy, Jz, NPARTICLE, active);
			cudaDeviceSynchronize();

		}
		saveFrame(outFile, radius, Px, Py, Pz, angle, Wx, Wy, Wz, NPARTICLE, active);
		saveSize(sizeFile, (float)frames / FRAMERATE, Px, Py, Pz, mass, NPARTICLE, active);

		if ((DATAPERIOD > 0) && (frames % (DATAPERIOD*FRAMERATE) == 0)) {

			int activeParticles = 0;
			for (int p = 0; p < NPARTICLE; p++) {
				if (active[p]) activeParticles ++;
			}

			// Mass distribution
			dataFileName = string(DATADIR) + string("mass_E") + to_string(realE0) + string("_") + to_string(DATAPERIOD * (frames/(DATAPERIOD*FRAMERATE))) + string("s.dat");
			ofstream mdFile (dataFileName.c_str(), ios::out | ios::trunc | ios::binary);
			mdFile.write((char *)(&activeParticles), sizeof(int));
			for (int p = 0; p < NPARTICLE; p++) {
				if (active[p]) mdFile.write((char *)(mass+p), sizeof(float));
			}
			mdFile.close();

			// Energy distribution
			auto Ep = [=] (int p) { return getSingleEnergy(active, mass, Px, Py, Pz, Vx, Vy, Vz, radius, Wx, Wy, Wz, NPARTICLE, p); };
			dataFileName = string(DATADIR) + string("energy_E") + to_string(realE0) + string("_") + to_string(DATAPERIOD * (frames/(DATAPERIOD*FRAMERATE))) + string("s.dat");
			ofstream edFile (dataFileName.c_str(), ios::out | ios::trunc | ios::binary);
			edFile.write((char *)(&activeParticles), sizeof(int));
			for (int p = 0; p < NPARTICLE; p++) {
				float curEp = Ep(p);
				if (active[p]) edFile.write((char *)(&curEp), sizeof(float));
			}
			edFile.close();
		}
		frames ++;
		if (!stopped) printProgress(frames, DURATION*FRAMERATE);
	}	
	cout << "\n";
	outFile.close();
	sizeFile.close();
	// saveProgress (outFileName, NPARTICLE, FRAMESTEP, active, radius, mass, Px, Py, Pz, angle, Wx, Wy, Wz, Vx, Vy, Vz);

	cudaFree(mass);
	cudaFree(radius);
	cudaFree(angle);

	cudaFree(Px);
	cudaFree(Py);
	cudaFree(Pz);

	cudaFree(Vx);
	cudaFree(Vy);
	cudaFree(Vz);

	cudaFree(Wx);
	cudaFree(Wy);
	cudaFree(Wz);

	cudaFree(Fx);
	cudaFree(Fy);
	cudaFree(Fz);

	cudaFree(Sx);
	cudaFree(Sy);
	cudaFree(Sz);

	cudaFree(Jx);
	cudaFree(Jy);
	cudaFree(Jz);

	cudaFree(active);
	return 0;
}