#include <iostream>
#include <fstream>
#include <string>
#include <signal.h>
#define FRAMERATE 60
#define EFACTOR 0.2
#define KFACTOR 0 // 0.005
#define SCHEMEORDER 4
#define ZEROTHRESHOLD 0.2
#define PI 3.14159265359
using namespace std;

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
		angle[p] = fmodf(angle[p] + sqrtf(Wx[p]*Wx[p] + Wy[p]*Wy[p] + Wz[p]*Wz[p]) * dta, 2*PI);

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
			float distance = sqrt(Dx*Dx + Dy*Dy + Dz*Dz);
			float Nx = Dx/distance, Ny = Dy/distance, Nz = Dz/distance;
			if (distance <= radius[p1] + radius[p2]) {
				float Cx = Nx * radius[p1], Cy = Ny * radius[p1], Cz = Nz * radius[p1];
				float Vrx = Vx[p1] - Vx[p2], Vry = Vy[p1] - Vy[p2], Vrz = Vz[p1] - Vz[p2];
				float nvr = Vrx*Nx + Vry*Ny + Vrz*Nz;
				if (abs(nvr) > ZEROTHRESHOLD) {

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
	srand(time(0));

	string outFileName;
	unsigned int DURATION = 60, FRAMESTEP = 10, NPARTICLE = 500; // Optional terminal paramenters IN THIS ORDER!
	if (argc > 1) outFileName = string(argv[argc-1]);
	else {
		cerr << "You must specify an output or save file.\nUsage: ./core [DURATION] [FRAMESTEP] [NPARTICLE] \"OutputFile.txt\"\nor\n./core \"ProgressData.dat\"" << endl;
		return 1;
	}
	bool resuming = outFileName.substr(outFileName.length()-4, 4).compare(string(".dat")) == 0;
	if (!resuming) {
		if (argc > 2) NPARTICLE = stoi(argv[argc-2]);
		if (argc > 3) FRAMESTEP = stoi(argv[argc-3]);
		if (argc > 4) DURATION = stoi(argv[argc-4]);
	}
	else {
		if (argc > 2) DURATION = stoi(argv[argc-2]);
	}
	if (DURATION == 0) DURATION = UINT_MAX;
	
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
		for (int p = 0; p < NPARTICLE; p++) {
			active[p] = true;
			angle[p] = 0;
			Wx[p] = random(-1, 1);
			Wy[p] = random(-1, 1);
			Wz[p] = random(-1, 1);
			mass[p] = random(0.5, 2);
			radius[p] = random(0.1, 0.2);
			float rx = random(-40, 40), ry = random(-40, 40), rz = random(-40, 40), rvn = random(-3, 1);
			Px[p] = rx;
			Py[p] = ry;
			Pz[p] = rz;
			float rn = sqrt(rx*rx + ry*ry + rz*rz);
			Vx[p] = (rx/rn) * rvn;
			Vy[p] = (ry/rn) * rvn;
			Vz[p] = (rz/rn) * rvn;
		}
	}
	for (int p = 0; p < NPARTICLE; p++) {
		Fx[p] = -KFACTOR * Px[p];
		Fy[p] = -KFACTOR * Py[p];
		Fz[p] = -KFACTOR * Pz[p];

		Sx[p] = Sy[p] = Sz[p] = Jx[p] = Jy[p] = Jz[p] = 0;
	}

	// Output file initialization
	ofstream outFile (outFileName.c_str(), ios::out | ios::binary | (resuming ? ios::app : ios::trunc));
	if (!outFile.is_open()) {
		cerr << "Error opening output file." << endl;
		return 2;
	}
	if (!resuming) outFile.write((char *)(&NPARTICLE), sizeof(unsigned int));

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
		frames ++;
		if (!stopped) printProgress(frames, DURATION*FRAMERATE);
	}	
	cout << "\n";
	outFile.close();
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