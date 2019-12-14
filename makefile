all: compile link run

compile:
	mkdir -p compiled
	cd compiled; \
	g++ -g -O2 -fstack-protector-strong -Wformat -Werror=format-security -fvisibility-inlines-hidden -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_THREAD_SAFE -D_REENTRANT \
	-c ../*.cpp

	cd compiled; \
	g++ -g -O2 -fstack-protector-strong -Wformat -Werror=format-security -fvisibility-inlines-hidden -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_THREAD_SAFE -D_REENTRANT \
	-c ../graphic/*.cpp

link:
	mkdir -p compiled
	cd compiled; \
	g++ -g -O2 -fstack-protector-strong -Wformat -Werror=format-security -fvisibility-inlines-hidden -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_THREAD_SAFE -D_REENTRANT \
	-o ../Main *.o -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-z,now \
	-lfltk_gl -lfltk -lX11 -pthread -lglut -lGLU -lGL

run:
	./Main

clear:
	rm -r compiled

cuda:
	cd CUDA; \
	nvcc --use_fast_math -o coreGPU core_GPU.cu;