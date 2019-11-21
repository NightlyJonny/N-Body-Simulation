#include <iostream>
#include <fstream>
#include <cmath>
#include <signal.h>
#define FRAMERATE 60
#define EFACTOR 0.2
#define KFACTOR 0 // 0.005
#define SCHEMEORDER 4
using namespace std;

void saveFrame (ofstream& outFile, float* Px, float* Py, int particleNumber) {
	for (int i = 0; i < particleNumber; i ++) {
		outFile << Px[i] << "," << Py[i] << "\t";
	}
	outFile << "\n";
}

float random (float min, float max) {
	// Useless comment so I can fold this function in Sublime Text :)
	return ((float)rand() / RAND_MAX) * (max-min) + min;
}

void printProgress (int currentFrame, int totalFrames) {
	float percentage = ((float)currentFrame / (float)totalFrames) * 100;
	cout << "\r[";
	for (int i = 0; i < 50; i++) {
		if (i == 24) cout << round(percentage) << " %";
		else if (i < round(percentage/2.0)) cout << "#";
		else cout << " ";
	}
	cout << "]" << flush;
}

void saveProgress(char* ofName, int NPARTICLE, int FRAMESTEP, float* mass, float* radius, float* Px, float* Py, float* Vx, float* Vy) {
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

__global__ void computeForces (float* Fx, float* Fy, float* Px, float* Py, float* mass, int NPARTICLE) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < NPARTICLE*NPARTICLE) {
		int p1 = p / NPARTICLE, p2 = p % NPARTICLE;
		if (p1 != p2) {
			float Dx = Px[p2]-Px[p1], Dy = Py[p2]-Py[p1];
			float distance2 = Dx*Dx + Dy*Dy, distance = sqrtf(distance2);
			float curFx = (Dx/distance) * mass[p1] * mass[p2] / distance2, curFy = (Dy/distance) * mass[p1] * mass[p2] / distance2;
			atomicAdd(Fx+p1, curFx);
			atomicAdd(Fy+p1, curFy);
		}
	}
}

__global__ void moveSystem (float* Fx, float* Fy, float* Px, float* Py, float* Vx, float* Vy, float* mass, float dtv, float dtp, int NPARTICLE) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < NPARTICLE) {
		atomicAdd(Vx+p, (Fx[p]/mass[p]) * dtv);
		atomicAdd(Vy+p, (Fy[p]/mass[p]) * dtv);
		atomicAdd(Px+p, Vx[p] * dtp);
		atomicAdd(Py+p, Vy[p] * dtp);
		// atomicAdd(Px+p, Vx[p] * dt + (Fx[p]/mass[p]) * dt*dt);
		// atomicAdd(Py+p, Vy[p] * dt + (Fy[p]/mass[p]) * dt*dt);
		Fx[p] = -KFACTOR * Px[p];
		Fy[p] = -KFACTOR * Py[p];
	}
}

__global__ void computeCollisions (float* Px, float* Py, float* Vx, float* Vy, float* Sx, float* Sy, float* Jx, float* Jy,float* mass, float* radius, int NPARTICLE) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < NPARTICLE*NPARTICLE) {
		int p1 = p / NPARTICLE, p2 = p % NPARTICLE;
		if (p1 != p2) {
			float Dx = Px[p1] - Px[p2], Dy = Py[p1] - Py[p2];
			float distance2 = Dx*Dx + Dy*Dy, distance = sqrt(distance2);
			float Nx = Dx/distance, Ny = Dy/distance;
			if (distance <= radius[p1] + radius[p2]) {

				float cSx = Nx * 1.0001*(radius[p1] + radius[p2]) - Dx, cSy = Ny * 1.0001*(radius[p1] + radius[p2]) - Dy;
				atomicAdd(Sx+p1, cSx / 2);
				atomicAdd(Sy+p1, cSy / 2);

				float cJr = ((EFACTOR+1)/(1/mass[p1] + 1/mass[p2])) * ((Vx[p1]-Vx[p2])*Nx + (Vy[p1]-Vy[p2])*Ny);
				atomicAdd(Jx+p1, -Nx * (cJr/mass[p1]));
				atomicAdd(Jy+p1, -Ny * (cJr/mass[p1]));
			}
		}
	}
}

__global__ void collisionResponse (float* Px, float* Py, float* Vx, float* Vy, float* Sx, float* Sy, float* Jx, float* Jy, int NPARTICLE) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < NPARTICLE) {
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
	int DURATION = 60, FRAMESTEP = 10, NPARTICLE = 100; // Optional terminal paramenters IN THIS ORDER!
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
	
	float *mass, *radius, *Px, *Py, *Vx, *Vy, *Fx, *Fy, *Sx, *Sy, *Jx, *Jy;
	cudaMallocManaged(&mass, NPARTICLE*sizeof(float));
	cudaMallocManaged(&radius, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Px, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Py, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Vx, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Vy, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Fx, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Fy, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Sx, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Sy, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Jx, NPARTICLE*sizeof(float));
	cudaMallocManaged(&Jy, NPARTICLE*sizeof(float));

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
			mass[p] = 10;
			radius[p] = 0.5;
			float rx = random(-200, 200), ry = random(-200, 200), rv = random(0, 20);
			Px[p] = rx;
			Py[p] = ry;
			float rxn = sqrt(rx*rx + ry*ry);
			Vx[p] = (rx/rxn) * rv;
			Vy[p] = (ry/rxn) * rv;
			Fx[p] = Fy[p] = 0;
		}
	}

	ofstream outFile (outFileName, ios::out | (resuming ? ios::app : ios::trunc));
	if (!outFile.is_open()) {
		cerr << "Error opening output file." << endl;
		return 2;
	}
	if (!resuming) outFile << radius[0] << "\n";

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

	int frames = 0;
	const float dt = 1.0/(FRAMERATE*FRAMESTEP);
	while ((frames < DURATION*FRAMERATE) && (!stopped)) {
		for (int i = 0; i < FRAMESTEP; i ++) {

			for (int s = 0; s < SCHEMEORDER; s ++) {
				computeForces <<<(NPARTICLE*NPARTICLE+255)/256, 256>>> (Fx, Fy, Px, Py, mass, NPARTICLE);
				cudaDeviceSynchronize();

				moveSystem <<<(NPARTICLE+255)/256, 256>>> (Fx, Fy, Px, Py, Vx, Vy, mass, cs[s]*dt, cd[s]*dt, NPARTICLE);
				cudaDeviceSynchronize();
			}

			computeCollisions <<<(NPARTICLE*NPARTICLE+255)/256, 256>>> (Px, Py, Vx, Vy, Sx, Sy, Jx, Jy, mass, radius, NPARTICLE);
			cudaDeviceSynchronize();

			collisionResponse <<<(NPARTICLE+255)/256, 256>>> (Px, Py, Vx, Vy, Sx, Sy, Jx, Jy, NPARTICLE);
			cudaDeviceSynchronize();

		}
		saveFrame(outFile, Px, Py, NPARTICLE);
		frames ++;
		if (!stopped) printProgress(frames, DURATION*FRAMERATE);
	}	
	cout << "\n";
	outFile.close();
	saveProgress(outFileName, NPARTICLE, FRAMESTEP, mass, radius, Px, Py, Vx, Vy);

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