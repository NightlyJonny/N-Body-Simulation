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
#include <bits/stdc++.h>

#include "graphic/Form.h"
#include "Simulation.h"

Form* form = nullptr;
Simulation* sim;


struct SettingsWidget {
	Fl_Choice* typeChoice;
	Fl_Input* outFileTxt;
	Fl_Input* durationTxt;
	Fl_Input* fsTxt;
	Fl_Input* pnTxt;
	Fl_Check_Button* outChk;
};

void idle_cb(void*) {

	form->getFrame()->redraw();
}

void start(bool gui, string outFile, unsigned int duration, unsigned int frameStep, unsigned int particleNumber) {

	sim = new Simulation(outFile, duration, frameStep, particleNumber, gui);
	std::thread heavyWork([]() { sim->core(); });

	//Create and execute the graphic part
	if (gui) {
		if(form != nullptr)
			form->setSimulation(sim);
		else
			form = new Form(sim);
		form->show();
		Fl::add_idle(idle_cb, 0);
		Fl::run();
		sim->stop();
	}

	heavyWork.join();
	sim->terminateSim();
}

void start_cb(Fl_Widget* o, void* settings) {
	SettingsWidget* sw = (SettingsWidget*)settings;
	bool gui = (sw->typeChoice->value() == 0) || (sw->typeChoice->value() == 3);
	
	unsigned int duration = stoi(sw->durationTxt->value());
	unsigned int frameStep = stoi(sw->fsTxt->value());
	unsigned int particleNumber = stoi(sw->pnTxt->value());
	string outFile = string(sw->outFileTxt->value());

	if(!gui && duration == 0) //The program can't start without gui if duration value is 0
		return;

	if(sw->typeChoice->value() == 2){
		string str = "./CUDA/coreGPU " + to_string(duration) + " " + to_string(frameStep) + " " + to_string(particleNumber) + " " + outFile;
		int i = system(str.c_str());
		return;
	}

	if(sw->typeChoice->value() == 3){
		start(gui, outFile, 0, 0, 0);
		return;
	}

	start(gui, outFile, duration, frameStep, particleNumber);
}

void quit_cb(Fl_Widget* o, void*) {

	exit(0);
}

void out_cb(Fl_Widget* o, void* txt) {

	Fl_Check_Button* check = (Fl_Check_Button*)o;
	Fl_Input* outFileTxt = (Fl_Input*)txt;

	if (check->value()) {
		outFileTxt->value("result.sim");
		outFileTxt->activate();
	}
	else {
		outFileTxt->value("/dev/null");
		outFileTxt->deactivate();
	}
}

void choice_cb(Fl_Widget* o, void* settings) {

	SettingsWidget* sw = (SettingsWidget*)settings;
	Fl_Choice *element = (Fl_Choice*)o;
	if (element->value() == 3) {
		sw->outFileTxt->label("   Input file:");
		sw->outFileTxt->activate();
		sw->outChk->value(1);
		sw->outChk->deactivate();

		sw->durationTxt->deactivate();
		sw->fsTxt->deactivate();
		sw->pnTxt->deactivate();
	}
	else {
		sw->outFileTxt->label("Output file:");
		sw->durationTxt->activate();
		sw->fsTxt->activate();
		sw->pnTxt->activate();
		sw->outChk->activate();
	}
}

int createWindow(){

	SettingsWidget* setWidgets = new SettingsWidget();

	Fl_Window win(400, 400, "Simulation Settings");

	win.begin();

		Fl_Return_Button startBtn (310, 360, 80, 30, "Start");
		Fl_Button quitBtn (10, 360, 80, 30, "&Quit");

		Fl_Choice typeChoice (140, 20, 100, 30, "Mode:");
		setWidgets->typeChoice = &typeChoice;
		typeChoice.add("FLTK live");
		typeChoice.add("Classic");
		typeChoice.add("CUDA");
		typeChoice.add("Viewer");
		typeChoice.value(0);

		Fl_Input outFileTxt (140, 70, 140, 30, "Output file:");
		outFileTxt.value("result.sim");
		setWidgets->outFileTxt = &outFileTxt;

		Fl_Check_Button outFileChk (290, 70, 120, 30, "Save on file");
		outFileChk.value(true);
		setWidgets->outChk = &outFileChk;

		Fl_Input durationTxt (140, 120, 140, 30, "Duration:");
		durationTxt.value("0");
		setWidgets->durationTxt = &durationTxt;

		Fl_Input fsTxt (140, 170, 140, 30, "Frame substeps:");
		fsTxt.value("10");
		setWidgets->fsTxt = &fsTxt;

		Fl_Input pnTxt (140, 220, 140, 30, "Particles number:");
		pnTxt.value("200");
		setWidgets->pnTxt = &pnTxt;

	win.end();
	startBtn.callback(start_cb, setWidgets);
	quitBtn.callback(quit_cb);
	outFileChk.callback(out_cb, &outFileTxt);
	typeChoice.callback(choice_cb, setWidgets);

	win.show();
	return Fl::run();
}

int main () {

	return createWindow();
}