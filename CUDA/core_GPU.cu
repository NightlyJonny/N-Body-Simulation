#include <iostream>
#include <fstream>
#include <signal.h>
#define FRAMERATE 60
#define EFACTOR 0.2
#define KFACTOR 0 // 0.005
#define SCHEMEORDER 4
#define ZEROTHRESHOLD 0.2
#define PI 3.14159265359
using namespace std;

void saveFrame (ofstream& outFile, float* radius, float* Px, float* Py, float* angle, int particleNumber, bool* active) {
	for (int i = 0; i < particleNumber; i++) {
		if (active[i])
			outFile << radius[i] << ":" << Px[i] << "," << Py[i] << << ",0;" << angle[i] << "\t";
		else
			outFile << "0" << "\t";
	}
	outFile << "\n";
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

void saveProgress (char* ofName, int NPARTICLE, int FRAMESTEP, float* mass, float* radius, float* Px, float* Py, float* Vx, float* Vy) {
	ofstream outFile ((string(ofName) + string(".dat")).c_str(), ios::out | ios::trunc | ios::binary);
	if (!outFile.is_open()) {
		cerr << "Error opening output file." << endl;
		return;
	}

	outFile.write((char*)(&NPARTICLE), sizeof(int));
	outFile.write((char*)(&FRAMESTEP), sizeof(int));
	for (int i = 0; i < NPARTICLE; i++) {
		outFile.write((char*)(mass + i), sizeof(float));
		outFile.write((char*)(radius + i), sizeof(float));
		outFile.write((char*)(Px + i), sizeof(float));
		outFile.write((char*)(Py + i), sizeof(float));
		outFile.write((char*)(Vx + i), sizeof(float));
		outFile.write((char*)(Vy + i), sizeof(float));
	}

	outFile.close();
}

bool stopped = false;
void sigHandler (int sig) {
	cout << flush << "\nRendering stopped, saving progress..." << endl;
	stopped = true;
}

__global__ void computeForces (float* Fx, float* Fy, float* Px, float* Py, float* mass, int NPARTICLE, bool* active) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < NPARTICLE*NPARTICLE) {
		int p1 = p / NPARTICLE, p2 = p % NPARTICLE;
		if ((p2 < NPARTICLE) && (p2 > p1) && active[p1] && active[p2]) {
			float Dx = Px[p2]-Px[p1], Dy = Py[p2]-Py[p1];
			float distance2 = Dx*Dx + Dy*Dy, distance = sqrtf(distance2);
			float curFx = (Dx/distance) * mass[p1] * mass[p2] / distance2, curFy = (Dy/distance) * mass[p1] * mass[p2] / distance2;
			atomicAdd(Fx+p1, curFx);
			atomicAdd(Fy+p1, curFy);
			atomicAdd(Fx+p2, -curFx);
			atomicAdd(Fy+p2, -curFy);
		}
	}
}

__global__ void moveSystem (float* Fx, float* Fy, float* Px, float* Py, float* Vx, float* Vy, float* angle, float* omega, float* mass, float dtv, float dtp, float dta, int NPARTICLE, bool* active) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < NPARTICLE && active[p]) {
		Vx[p] += (Fx[p]/mass[p]) * dtv;
		Vy[p] += (Fy[p]/mass[p]) * dtv;
		Px[p] += Vx[p] * dtp;
		Py[p] += Vy[p] * dtp;
		angle[p] = fmodf(angle[p] + omega[p] * dta, 2*PI);
		Fx[p] = -KFACTOR * Px[p];
		Fy[p] = -KFACTOR * Py[p];
	}
}

__global__ void computeCollisions (float* Px, float* Py, float* Vx, float* Vy, float* omega, float* Sx, float* Sy, float* Jx, float* Jy,float* mass, float* radius, int NPARTICLE, bool* active) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < NPARTICLE*NPARTICLE) {
		int p1 = p / NPARTICLE, p2 = p % NPARTICLE;
		if ((p2 < NPARTICLE) && (p2 > p1) && active[p1] && active[p2]) {
			float Dx = Px[p1] - Px[p2], Dy = Py[p1] - Py[p2];
			float distance = sqrt(Dx*Dx + Dy*Dy);
			float Nx = Dx/distance, Ny = Dy/distance;
			if (distance <= radius[p1] + radius[p2]) {
				float Cx = Nx * radius[p1], Cy = Ny * radius[p1];
				float Vrx = Vx[p1] - Vx[p2], Vry = Vy[p1] - Vy[p2];
				float nvr = Vrx*Nx + Vry*Ny;
				if (abs(nvr) > ZEROTHRESHOLD) {

					// Collision
					float cSx = Nx * 1.0001*(radius[p1] + radius[p2]) - Dx, cSy = Ny * 1.0001*(radius[p1] + radius[p2]) - Dy;
					atomicAdd( Sx+p1, cSx * mass[p1] / (mass[p1] + mass[p2]) );
					atomicAdd( Sy+p1, cSy * mass[p1] / (mass[p1] + mass[p2]) );
					atomicAdd( Sx+p2, -cSx * mass[p2] / (mass[p1] + mass[p2]) );
					atomicAdd( Sy+p2, -cSy * mass[p2] / (mass[p1] + mass[p2]) );

					float cJr = ((EFACTOR+1)/(1/mass[p1] + 1/mass[p2])) * nvr;
					atomicAdd( Jx+p1, -Nx * (cJr/mass[p1]) );
					atomicAdd( Jy+p1, -Ny * (cJr/mass[p1]) );
					atomicAdd( Jx+p2, Nx * (cJr/mass[p2]) );
					atomicAdd( Jy+p2, Ny * (cJr/mass[p2]) );
				}
				else {
					// Fusion
					Px[p1] = (Px[p1] * mass[p1] + Px[p2] * mass[p2]) / (mass[p1] + mass[p2]);
					Py[p1] = (Py[p1] * mass[p1] + Py[p2] * mass[p2]) / (mass[p1] + mass[p2]);

					Vx[p1] = (Vx[p1] * mass[p1] + Vx[p2] * mass[p2]) / (mass[p1] + mass[p2]);
					Vy[p1] = (Vy[p1] * mass[p1] + Vy[p2] * mass[p2]) / (mass[p1] + mass[p2]);

					float newRadPow2 = radius[p1]*radius[p1] + radius[p2]*radius[p2];
					omega[p1] = (mass[p1] * radius[p1]*radius[p1] * omega[p1] + mass[p2]* radius[p2]*radius[p2] * omega[p2] + 2 * mass[p2] * (Cx*Vry - Vrx*Cy)) / ((mass[p1] + mass[p2]) * newRadPow2);
					mass[p1] += mass[p2];
					radius[p1] = sqrt(newRadPow2);

					active[p2] = false;
				}

			}
		}
	}
}

__global__ void collisionResponse (float* Px, float* Py, float* Vx, float* Vy, float* Sx, float* Sy, float* Jx, float* Jy, int NPARTICLE, bool* active) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < NPARTICLE && active[p]) {
		Px[p] += Sx[p];
		Py[p] += Sy[p];
		Vx[p] += Jx[p];
		Vy[p] += Jy[p];
		Sx[p] = Sy[p] = Jx[p] = Jy[p] = 0;
	}
}

int main(int argc, char** argv) {
	signal(SIGINT, sigHandler);
	char* outFileName;
	unsigned int DURATION = 60, FRAMESTEP = 10, NPARTICLE = 500; // Optional terminal paramenters IN THIS ORDER!
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
	if (DURATION == 0) DURATION = UINT_MAX;
	
	float *mass, *radius, *Px, *Py, *Vx, *Vy, *angle, *omega, *Fx, *Fy, *Sx, *Sy, *Jx, *Jy;
	bool *active;
	cudaMallocManaged(&mass, NPARTICLE*sizeof(float));
	cudaMallocManaged(&radius, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Px, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Py, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Vx, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Vy, NPARTICLE*sizeof(float));
	cudaMallocManaged(&angle, NPARTICLE*sizeof(float));
	cudaMallocManaged(&omega, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Fx, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Fy, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Sx, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Sy, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Jx, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Jy, NPARTICLE*sizeof(float));
	cudaMallocManaged(&active, NPARTICLE*sizeof(bool));

	// Initialization objects
	if (resuming) {
		cout << "Save file specified, restoring progress..." << endl;
		ifstream inFile (outFileName, ios::in | ios::binary);
		if (!inFile.is_open()) {
			cerr << "Error opening input file." << endl;
			return 2;
		}

		inFile.read((char*)&NPARTICLE, sizeof(int));
		inFile.read((char*)&FRAMESTEP, sizeof(int));
		for (int p = 0; p < NPARTICLE; p++) {
			inFile.read((char*)(mass+p), sizeof(float));
			inFile.read((char*)(radius+p), sizeof(float));
			inFile.read((char*)(Px+p), sizeof(float));
			inFile.read((char*)(Py+p), sizeof(float));
			inFile.read((char*)(Vx+p), sizeof(float));
			inFile.read((char*)(Vy+p), sizeof(float));

			if (isnan(mass[p])) cerr << "Mass read problem :(" << endl;
			if (isnan(radius[p])) cerr << "Radius read problem :(" << endl;
			if (isnan(Px[p])) cerr << "Px read problem :(" << endl;
			if (isnan(Py[p])) cerr << "Py read problem :(" << endl;
			if (isnan(Vx[p])) cerr << "Vx read problem :(" << endl;
			if (isnan(Vy[p])) cerr << "Vy read problem :(" << endl;
		}
		inFile.close();
		strcpy(outFileName, string(outFileName).substr(0, string(outFileName).length()-4).c_str());
	}
	else {
		for (int p = 0; p < NPARTICLE; p++) {
			active[p] = true;
			angle[p] = 0;
			omega[p] = random(-1, 1);
			mass[p] = random(0.5, 2);
			radius[p] = random(0.1, 0.2);
			float rx = random(-100, 100), ry = random(-100, 100), rvn = random(-3, 1), rvt = random(0, 7);
			Px[p] = rx;
			Py[p] = ry;
			float rn = sqrt(rx*rx + ry*ry);
			Vx[p] = (rx/rn) * rvn - (ry/rn) * rvt;
			Vy[p] = (ry/rn) * rvn + (rx/rn) * rvt;

			Fx[p] = -KFACTOR * Px[p];
			Fy[p] = -KFACTOR * Py[p];
			Sx[p] = Sy[p] = Jx[p] = Jy[p] = 0;
		}
	}

	ofstream outFile (outFileName, ios::out | (resuming ? ios::app : ios::trunc));
	if (!outFile.is_open()) {
		cerr << "Error opening output file." << endl;
		return 2;
	}
	if (!resuming) outFile << NPARTICLE << "\n";

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
				computeForces <<<(NPARTICLE*NPARTICLE+255)/256, 256>>> (Fx, Fy, Px, Py, mass, NPARTICLE, active);
				cudaDeviceSynchronize();

				moveSystem <<<(NPARTICLE+255)/256, 256>>> (Fx, Fy, Px, Py, Vx, Vy, angle, omega, mass, cs[s]*dt, cd[s]*dt, dt/SCHEMEORDER, NPARTICLE, active);
				cudaDeviceSynchronize();
			}

			computeCollisions <<<(NPARTICLE*NPARTICLE+255)/256, 256>>> (Px, Py, Vx, Vy, omega, Sx, Sy, Jx, Jy, mass, radius, NPARTICLE, active);
			cudaDeviceSynchronize();

			collisionResponse <<<(NPARTICLE+255)/256, 256>>> (Px, Py, Vx, Vy, Sx, Sy, Jx, Jy, NPARTICLE, active);
			cudaDeviceSynchronize();

		}
		saveFrame(outFile, radius, Px, Py, angle, NPARTICLE, active);
		frames ++;
		if (!stopped) printProgress(frames, DURATION*FRAMERATE);
	}	
	cout << "\n";
	outFile.close();
	// saveProgress(outFileName, NPARTICLE, FRAMESTEP, mass, radius, Px, Py, Vx, Vy); // STILL NOT WORKING :(

	cudaFree(mass);
	cudaFree(radius);
	cudaFree(Px);
	cudaFree(Py);
	cudaFree(Vx);
	cudaFree(Vy);
	cudaFree(Fx);
	cudaFree(Fy);
	return 0;
}