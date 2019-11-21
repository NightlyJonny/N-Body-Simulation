#include <iostream>
#include <fstream>
#include <math.h>
#include <signal.h>
#include <omp.h>
#define FRAMERATE 60
#define EFACTOR 0.2
#define KFACTOR 0.01

void saveFrame (std::ofstream& outFile, double* Px, double* Py, int particleNumber) {
	for (int i = 0; i < particleNumber; i ++) {
		outFile << Px[i] << "," << Py[i] << "\t";
	}
	outFile << "\n";
}

double random (double min, double max) {
	return ((double)rand() / RAND_MAX) * (max-min) + min;
}

void printProgress (int currentFrame, int totalFrames) {
	double percentage = ((double)currentFrame / (double)totalFrames) * 100;
	std::cout << "\r[";
	for (int i = 0; i < 50; i++) {
		if (i == 24) std::cout << std::round(percentage) << " %";
		else if (i < round(percentage/2.0)) std::cout << "#";
		else std::cout << " ";
	}
	std::cout << "]" << std::flush;
}

bool stopped = false;
void sigHandler (int sig) {
	std::cout << std::flush << "\nRendering stopped, saving progress..." << std::endl;
	stopped = true;
}

int main(int argc, char** argv) {
	signal(SIGINT, sigHandler);
	char* outFileName;
	int DURATION = 60, FRAMESTEP = 10, NPARTICLE = 100; // Optional terminal paramenters IN THIS ORDER!
	if (argc > 1) outFileName = argv[argc-1];
	else {
		std::cerr << "You must specify an output file.\nUsage: ./core [DURATION] [FRAMESTEP] [NPARTICLE] \"OutputFile.txt\"\nor\n./core \"ProgressData.dat\"" << std::endl;
		return 1;
	}
	if (argc > 2) NPARTICLE = std::stoi(argv[argc-2]);
	if (argc > 3) FRAMESTEP = std::stoi(argv[argc-3]);
	if (argc > 4) DURATION = std::stoi(argv[argc-4]);
	
	double* mass = new double[NPARTICLE];
	double* radius = new double[NPARTICLE];
	double* Px = new double[NPARTICLE];
	double* Py = new double[NPARTICLE];
	double* Vx = new double[NPARTICLE];
	double* Vy = new double[NPARTICLE];
	double* Fx = new double[NPARTICLE];
	double* Fy = new double[NPARTICLE];
	double* Sx = new double[NPARTICLE];
	double* Sy = new double[NPARTICLE];
	double* Jx = new double[NPARTICLE];
	double* Jy = new double[NPARTICLE];
	for (int p = 0; p < NPARTICLE; p++) {
		mass[p] = 1;
		radius[p] = 0.1;
		double rx = random(-10, 10), ry = random(-10, 10), rv = random(0, 4);
		Px[p] = rx;
		Py[p] = ry;
		double rxn = sqrt(rx*rx + ry*ry);
		Vx[p] = (rx/rxn) * rv;
		Vy[p] = (ry/rxn) * rv;
	}

	std::ofstream outFile (outFileName, std::ios::out | std::ios::trunc);
	if (!outFile.is_open()) {
		std::cerr << "Error opening output file." << std::endl;
		return 2;
	}

	int frames = 0;
	double Dx, Dy, distance2, distance;
	while ((frames < DURATION*FRAMERATE) && (!stopped)) {
		for (int i = 0; i < FRAMESTEP; i ++) {
			// Particle movement
			for (int p1 = 0; p1 < NPARTICLE; p1++) {
				for (int p2 = p1+1; p2 < NPARTICLE; p2++) {
					Dx = Px[p2]-Px[p1], Dy = Py[p2]-Py[p1];
					distance2 = Dx*Dx + Dy*Dy, distance = sqrt(distance2);
					double curFx = (Dx/distance) * mass[p1] * mass[p2] / distance2, curFy = (Dy/distance) * mass[p1] * mass[p2] / distance2;
					Fx[p1] = Fx[p1] + curFx;
					Fy[p1] = Fy[p1] + curFy;
					Fx[p2] = Fx[p2] - curFx;
					Fy[p2] = Fy[p2] - curFy;
				}
			}
			for (int p = 0; p < NPARTICLE; ++p) {
				double dt = 1.0/(FRAMERATE*FRAMESTEP);
				Px[p] += Vx[p] * dt + (Fx[p]/mass[p]) * dt*dt;
				Py[p] += Vy[p] * dt + (Fy[p]/mass[p]) * dt*dt;
				Vx[p] += (Fx[p]/mass[p]) * dt;
				Vy[p] += (Fy[p]/mass[p]) * dt;
				Fx[p] = -KFACTOR * Px[p];
				Fy[p] = -KFACTOR * Py[p];
			}

			// Collision detection (discrete)
			for (int p1 = 0; p1 < NPARTICLE; p1++) {
				for (int p2 = 0; p2 < NPARTICLE; p2++) {
					if (p1 != p2) {
						Dx = Px[p1] - Px[p2], Dy = Py[p1] - Py[p2];
						distance2 = Dx*Dx + Dy*Dy, distance = sqrt(distance2);
						double Nx = Dx/distance, Ny = Dy/distance;
						if (distance <= radius[p1] + radius[p2]) {
							// Collision response
							double cSx = Nx * 1.0001*(radius[p1] + radius[p2]) - Dx, cSy = Ny * 1.0001*(radius[p1] + radius[p2]) - Dy;
							Sx[p1] += cSx / 2;
							Sy[p1] += cSy / 2;

							double cJr = ((EFACTOR+1)/(1/mass[p1] + 1/mass[p2])) * ((Vx[p1]-Vx[p2])*Nx + (Vy[p1]-Vy[p2])*Ny);
							Jx[p1] -= Nx * (cJr/mass[p1]);
							Jy[p1] -= Ny * (cJr/mass[p1]);
						}
					}
				}
			}
			for (int p = 0; p < NPARTICLE; p++) {
				Px[p] += Sx[p];
				Py[p] += Sy[p];
				Vx[p] += Jx[p];
				Vy[p] += Jy[p];
				Sx[p] = Sy[p] = Jx[p] = Jy[p] = 0;
			}
		}
		saveFrame(outFile, Px, Py, NPARTICLE);
		frames ++;
		if (!stopped) printProgress(frames, DURATION*FRAMERATE);
	}	
	std::cout << "\n";
	outFile.close();

	delete[] mass;
	delete[] radius;
	delete[] Px;
	delete[] Py;
	delete[] Vx;
	delete[] Vy;
	return 0;
}
