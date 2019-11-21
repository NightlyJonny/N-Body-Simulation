#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <FL/Fl.H>
#include <thread>
#include "graphic/Form.h"
#include "Simulation.h"

Form* form;
Simulation* sim;

void idle_cb(void*) {

	form->getFrame()->redraw();
}

int main(int argc, char** argv) {

	// If this variable is on false the GUI is disabled
	bool gui = true;

	cout << "Simulation started" << endl;
	sim = new Simulation(argc, argv);

	std::thread heavyWork([]() {sim->core(); });

	//Create and execute the graphic part
	if (gui) {
		form = new Form(sim);
		Fl::add_idle(idle_cb, 0);
		Fl::run();
		sim->stop();
	}

	heavyWork.join();
	sim->terminateSim();

	return 0;
}