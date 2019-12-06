#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Return_Button.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Output.H>
#include <FL/Fl_Choice.H>
#include <thread>
#include "graphic/Form.h"
#include "Simulation.h"

Form* form;
Simulation* sim;
bool firstRun = true;

struct SettingsWidget {
	Fl_Choice* typeChoice;
	Fl_Input* outFileTxt;
	Fl_Input* durationTxt;
	Fl_Input* fsTxt;
	Fl_Input* pnTxt;
};

void idle_cb(void*) {

	form->getFrame()->redraw();
}

void start(bool gui, string outFile, unsigned int duration, unsigned int frameStep, unsigned int particleNumber) {

	sim = new Simulation(outFile, duration, frameStep, particleNumber, gui);
	std::thread heavyWork([]() {sim->core(); });

	//Create and execute the graphic part
	if (gui) {
		form = new Form(sim, firstRun);
		firstRun = false;
		Fl::add_idle(idle_cb, 0);
		Fl::run();
		sim->stop();
	}

	heavyWork.join();
	sim->terminateSim();
}

void start_cb(Fl_Widget* o, void* settings) {
	SettingsWidget* sw = (SettingsWidget*)settings;
	bool gui = (sw->typeChoice->value() == 0);
	unsigned int duration = stoi(sw->durationTxt->value());
	unsigned int frameStep = stoi(sw->fsTxt->value());
	unsigned int particleNumber = stoi(sw->pnTxt->value());
	string outFile = string(sw->outFileTxt->value());

	start(gui, outFile, duration, frameStep, particleNumber);
}

void quit_cb(Fl_Widget* o, void*) {

	exit(0);
}

void out_cb(Fl_Widget* o, void* txt) {

	Fl_Check_Button* check = (Fl_Check_Button*)o;
	Fl_Input* outFileTxt = (Fl_Input*)txt;

	if (check->value()) {
		outFileTxt->value("outfile.txt");
		outFileTxt->activate();
	}
	else {
		outFileTxt->value("/dev/null");
		outFileTxt->deactivate();
	}
}

int main () {

	Fl_Window win(400, 400, "Simulation Settings");
	SettingsWidget setWidgets;

	win.begin();

		Fl_Return_Button startBtn (310, 360, 80, 30, "Start");
		Fl_Button quitBtn (10, 360, 80, 30, "Quit");

		Fl_Choice typeChoice (130, 20, 100, 30, "Simulation type:");
		setWidgets.typeChoice = &typeChoice;
		typeChoice.add("FLTK live");
		typeChoice.add("Classic");
		typeChoice.value(0);

		Fl_Input outFileTxt (90, 70, 140, 30, "Output file:");
		outFileTxt.value("outfile.txt");
		setWidgets.outFileTxt = &outFileTxt;
		Fl_Check_Button outFileChk (240, 70, 120, 30, "Save on file");
		outFileChk.value(true);

		Fl_Input durationTxt (78, 120, 140, 30, "Duration:");
		durationTxt.value("0");
		setWidgets.durationTxt = &durationTxt;

		Fl_Input fsTxt (120, 170, 140, 30, "Frame susteps:");
		fsTxt.value("10");
		setWidgets.fsTxt = &fsTxt;

		Fl_Input pnTxt (135, 220, 140, 30, "Particles number:");
		pnTxt.value("200");
		setWidgets.pnTxt = &pnTxt;

	win.end();
	startBtn.callback(start_cb, &setWidgets);
	quitBtn.callback(quit_cb);
	outFileChk.callback(out_cb, &outFileTxt);

	win.show();
	return Fl::run();
}