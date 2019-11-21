#include "Form.h"

#define SCREEN_WIDTH  1200
#define SCREEN_HEIGHT 800

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

		scene = new Frame(border, border, SCREEN_WIDTH, SCREEN_HEIGHT, 0);

		timeText = new Fl_Output(border + SCREEN_WIDTH / 2 - 100, h_est - 40 - border, 200, 40, "");
		timeText->box(FL_UP_BOX);
		timeText->labelsize(20);
		timeText->align(FL_ALIGN_RIGHT);

	form->end();
	form->show();
	scene->show();
}

//This update the time label
void Form::updateTime(float nextTime) {
	
	int tot = FRAMERATE;

	char text[20];
	int car = sprintf(text, "Time: %3.2f/%d", nextTime, tot);

	timeText->value(text, car);
	timeText->show();
}

void Form::debugText(const char* text) {

	timeText->value(text);
	timeText->show();
}

Frame* Form::getFrame(){

	return Form::scene;
}