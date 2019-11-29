#include "Form.h"

//Declaration of static members
Fl_Window* Form::form;
Frame* Form::scene;
Simulation* Form::sim;

Form::Form(Simulation* sim) {
	Form::sim = sim;
	CreateMyWindow();
	getFrame()->setSimulation(sim);
}

void Form::CreateMyWindow(void) {
	int w_est, h_est;
	int border = 5;
	int bottomGui = 40 + border;

	w_est = border + SCREEN_WIDTH + border;
	h_est = border + SCREEN_HEIGHT + bottomGui + border;

	form = new Fl_Window(w_est, h_est, "N-Body Simulation");
		// form->fullscreen();
		form->size_range(SCREEN_WIDTH, SCREEN_HEIGHT, 3840, 2160);
		scene = new Frame(border, border, SCREEN_WIDTH, SCREEN_HEIGHT, 0);
		form->resizable(scene);

		timeText = new Fl_Output(200, h_est - 40 - border, 150, 40, "");
		timeText->box(FL_UP_BOX);
		timeText->labelsize(20);
		timeText->align(FL_ALIGN_RIGHT);

		debugText = new Fl_Output(border, h_est - 40 - border, 150, 40, "");
		debugText->box(FL_UP_BOX);
		debugText->labelsize(20);
		debugText->align(FL_ALIGN_RIGHT);

	form->end();
	form->show();
	scene->show();
}

//This update the time label
void Form::updateTime(float nextTime) {
	
	unsigned int tot = sim->getDuration();

	char text[20];
	int car;
	if (tot > 0) {
		car = sprintf(text, "Time: %3.2f/%d", nextTime, tot);
	}
	else {
		car = sprintf(text, "Time: %3.2f", nextTime);
	}

	timeText->value(text, car);
	timeText->show();
}

void Form::updateDebug(string text) {

	debugText->value(text.c_str());
	debugText->show();
}

Frame* Form::getFrame(){

	return Form::scene;
}