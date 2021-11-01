all: task1 task2 task3 task4 task5 task6

task1: Task1.cpp
	mpic++ -O2 -march=native Task1.cpp -o Task1
task2: Task2.cpp
	mpic++ -O2 -march=native Task2.cpp -o Task2
task3: Task3.cpp
	mpic++ -O2 -march=native Task3.cpp -o Task3
task4: Task4.cpp
	mpic++ -O2 -march=native Task4.cpp -o Task4
task5: Task5.cpp
	mpic++ -O2 -march=native Task5.cpp -o Task5
task6: Task6.cpp
	mpic++ -O2 -march=native Task6.cpp -o Task6
clean: 
	rm -f Task1
	rm -f Task2
	rm -f Task3
	rm -f Task4
	rm -f Task5
	rm -f Task6	