#ifndef FORM_HPP
#define FORM_HPP

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Multiline_Output.H>
#include <string>
#include "Frame.h"

class Form{
private:
	static Fl_Window* form;
	static Frame* scene;

	Fl_Output* timeText;
	Fl_Output* debugText;
	Fl_Output* infoText;

	int floatToInt(float num, int afterComma) { return round(num * pow(10, afterComma)); }
	void CreateMyWindow();
public:
	Simulation* sim;
	Form(Simulation*);
	
	void setSimulation(Simulation*);
	void updateTime(float);
	void updateDebug(string);
	void updateInfo(string text);

	void show();

	Frame* getFrame();
};

#endif // FORM_HPP
