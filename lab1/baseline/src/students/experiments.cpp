#include <fstream>

#include "../utils/Timer.hpp"
#include "../utils/Matrix.hpp"
#include "../experiments.hpp"
#include "../matmult.hpp"

void generateHeader(std::stringstream &ss,
                    const std::vector<std::string> &headers,
                    const std::vector<std::string> &repeated_headers,
                    unsigned int repeats) {
  for (const auto &header : headers) {
    ss << std::setw(15) << (header + ",");
  }
  for (const auto &header : repeated_headers) {
    for (unsigned int r = 0; r < repeats; r++) {
      ss << std::setw(15) << (header + " " + std::to_string(r) + ",");
    }
  }
  ss << std::endl;
}

void dump(std::stringstream &ss, std::ostream &s0, std::ostream &s1) {
  s0 << ss.str();
  s1 << ss.str();
  ss.str("");
}

void runVectorExperiment(unsigned int min, unsigned int max, unsigned int repeats, std::string file_out) {
  std::cout << "Vector inner product benchmark." << std::endl;
  // Attempt to open the file for writing
  std::ofstream fos(file_out);
  if (!fos.good()) {
    std::cerr << "Could not open file " + file_out << std::endl;
    std::exit(-1);
  }
  // Generate a stringstream to write output to both stdout and a file
  std::stringstream ss;

  // Create a timer used for wall-clock time measurements
  Timer t;

  // Dump a header
  generateHeader(ss, {"Matrix size"}, {"Float", "Double"}, repeats);
  dump(ss, fos, std::cout);

  // Iterate over each experiment
  for (unsigned int e = min; e < max; e++) {
    auto mat_rows = 1ul << e;
    auto mat_cols = 1ul;


    ss << std::setw(15) << (std::to_string(mat_rows) + ",") << std::flush;

    // Create the matrices
    auto mat_a = Matrix<float>(mat_cols, mat_rows); // Make a matrix
    auto mat_b = Matrix<float>(mat_rows, mat_cols); // And another one, transposed.
    auto mat_c = Matrix<double>(mat_cols, mat_rows); // And another one, doubles.
    auto mat_d = Matrix<double>(mat_rows, mat_cols); // And another one, transposed.

    // Randomize their contents
    mat_a.randomize();
    mat_b.randomize();
    mat_c.randomize();
    mat_d.randomize();

    // Dump the initialization output
    dump(ss, fos, std::cout);

    // Repeat the floats experiment repeats times.
    for (unsigned int r = 0; r < repeats; r++) {
      t.start(); // Start the timer.
      // Multiply the matrices
      auto mat_result = Matrix<float>::multiply(mat_a, mat_b);
      t.stop(); // Stop the timer.
      t.report(ss);

      // Dump the repeat outcome
      dump(ss, fos, std::cout);
    }

    // Repeat the doubles experiment repeats times.
    for (unsigned int r = 0; r < repeats; r++) {
      t.start();
      auto mat_result = Matrix<double>::multiply(mat_c, mat_d);
      t.stop();
      t.report(ss, r == (repeats - 1));
      dump(ss, fos, std::cout);
    }
  }
}

void runMatrixExperiment(unsigned int min, unsigned int max, unsigned int repeats, std::string file_out) {
  std::cout << "Matrix benchmark." << std::endl;
  // Attempt to open the file for writing
  std::ofstream fos(file_out);
  if (!fos.good()) {
    std::cerr << "Could not open file " + file_out << std::endl;
    std::exit(-1);
  }
  // Generate a stringstream to write output to both stdout and a file
  std::stringstream ss;

  // Create a timer used for wall-clock time measurements
  Timer t;

  // Dump a header
  generateHeader(ss, { "Matrix size"}, {"Float Speedup", "Double Speedup"}, repeats);
  dump(ss, fos, std::cout);

  // Iterate over each experiment
  for (unsigned int e = min; e < max; e++) {
    // In this experiment, we use powers of 2 as the problem size.
    // Not that that is not always necessary. You may also linearly grow the problem size.
    // Shift a long value of 1 left by e, which is the same as 2^e, to obtain the matrix dimension
    auto mat_rows = 1ul << e;
    // Number of columns is 1 for now, because we just want to calculate the inner product.
    auto mat_cols = 1ul << e;

    // Print experiment number

    // Print the problem size
    ss << std::setw(15) << (std::to_string(mat_rows) + ",") << std::flush;

    // Create the matrices
    auto mat_a = Matrix<float>(mat_cols, mat_rows); // Make a matrix
    auto mat_b = Matrix<float>(mat_rows, mat_cols); // And another one, transposed.
    auto mat_c = Matrix<double>(mat_cols, mat_rows); // And another one, doubles.
    auto mat_d = Matrix<double>(mat_rows, mat_cols); // And another one, transposed.
    mat_a.randomize();
    mat_b.randomize();
    mat_c.randomize();
    mat_d.randomize();

    // Dump the initialization output
    dump(ss, fos, std::cout);

    for (unsigned int r = 0; r < repeats; r++) {
      t.start(); // Start the timer.
      // Multiply the matrices
      auto mat_result = Matrix<float>::multiply(mat_a, mat_b);
      t.stop(); // Stop the timer.
      t.report(ss);

      // Dump the repeat outcome
      dump(ss, fos, std::cout);
    }

    // Repeat the doubles experiment repeats times.
    for (unsigned int r = 0; r < repeats; r++) {
      t.start();
      auto mat_result = Matrix<double>::multiply(mat_c, mat_d);
      t.stop();
      t.report(ss, r == (repeats - 1));
      dump(ss, fos, std::cout);
    }
  }
}

void runMatrixExperimentSIMD(unsigned int min, unsigned int max, unsigned int repeats, std::string file_out) {
  std::cout << "SIMD benchmark." << std::endl;
  std::ofstream fos(file_out);
  if (!fos.good()) {
    std::cerr << "Could not open file " + file_out << std::endl;
    std::exit(-1);
  }
  std::stringstream ss;
  Timer t;
  generateHeader(ss, {"Matrix size"}, {"Float", "Double"}, repeats);
  dump(ss, fos, std::cout);
  for (unsigned int e = min; e < max; e++) {
    auto mat_rows = 1ul << e;
    auto mat_cols = 1ul << e;
    ss << std::setw(15) << (std::to_string(mat_rows) + ",") << std::flush;
    auto mat_a = Matrix<float>(mat_cols, mat_rows); // Make a matrix
    auto mat_b = Matrix<float>(mat_rows, mat_cols); // And another one, transposed.
    auto mat_c = Matrix<double>(mat_cols, mat_rows); // And another one, doubles.
    auto mat_d = Matrix<double>(mat_rows, mat_cols); // And another one, transposed.
    mat_a.randomize();
    mat_b.randomize();
    mat_c.randomize();
    mat_d.randomize();
    dump(ss, fos, std::cout);

    for (unsigned int r = 0; r < repeats; r++) {
      t.start(); // Start the timer.
      auto mat_r1 = Matrix<float>::multiply(mat_a, mat_b);
      t.stop(); // Stop the timer.
      double t1 = t.seconds();
      
      t.start(); // Start the timer.
      auto mat_r2 = multiplyMatricesSIMD(mat_a, mat_b);
      t.stop(); // Stop the timer.
      double t2 = t.seconds();
	
      std::cout << std::setprecision(3) << std::setw(14) << (t1/t2) << ",";
      dump(ss, fos, std::cout);
    }

    for (unsigned int r = 0; r < repeats; r++) {
      t.start(); // Start the timer.
      auto mat_r3 = Matrix<double>::multiply(mat_c, mat_d);
      t.stop(); // Stop the timer.
      double t1 = t.seconds();
      
      t.start(); // Start the timer.
      auto mat_r4 = multiplyMatricesSIMD(mat_c, mat_d);
      t.stop(); // Stop the timer.
      double t2 = t.seconds();
	
      std::cout << std::setprecision(3) << std::setw(14) << (t1/t2) << ",";
      dump(ss, fos, std::cout);
    }
      std::cout << " " << std::endl;
  }
}

void runMatrixExperimentOMP(unsigned int min,
                            unsigned int max,
                            unsigned int threads,
                            unsigned int repeats,
                            std::string file_out) {
  std::cout << "OMP benchmark." << std::endl;
  std::ofstream fos(file_out);
  if (!fos.good()) {
    std::cerr << "Could not open file " + file_out << std::endl;
    std::exit(-1);
  }
  std::stringstream ss;
  Timer t;
  generateHeader(ss, {"Matrix size"}, {"Float", "Double"}, repeats);
  dump(ss, fos, std::cout);
  for (unsigned int e = min; e < max; e++) {
    auto mat_rows = 1ul << e;
    auto mat_cols = 1ul << e;
    ss << std::setw(15) << (std::to_string(mat_rows) + ",") << std::flush;
    auto mat_a = Matrix<float>(mat_cols, mat_rows); // Make a matrix
    auto mat_b = Matrix<float>(mat_rows, mat_cols); // And another one, transposed.
    auto mat_c = Matrix<double>(mat_cols, mat_rows); // And another one, doubles.
    auto mat_d = Matrix<double>(mat_rows, mat_cols); // And another one, transposed.
    mat_a.randomize();
    mat_b.randomize();
    mat_c.randomize();
    mat_d.randomize();
    dump(ss, fos, std::cout);

    for (unsigned int r = 0; r < repeats; r++) {
      t.start(); // Start the timer.
      auto mat_r1 = Matrix<float>::multiply(mat_a, mat_b);
      t.stop(); // Stop the timer.
      double t1 = t.seconds();
      
      t.start(); // Start the timer.
      auto mat_r2 = multiplyMatricesOMP(mat_a, mat_b,threads);
      t.stop(); // Stop the timer.
      double t2 = t.seconds();
	
      std::cout << std::setprecision(3) << std::setw(14) << (t1/t2) << ",";
      dump(ss, fos, std::cout);
    }

    for (unsigned int r = 0; r < repeats; r++) {
      t.start(); // Start the timer.
      auto mat_r3 = Matrix<double>::multiply(mat_c, mat_d);
      t.stop(); // Stop the timer.
      double t1 = t.seconds();
      
      t.start(); // Start the timer.
      auto mat_r4 = multiplyMatricesOMP(mat_c, mat_d,threads);
      t.stop(); // Stop the timer.
      double t2 = t.seconds();
	
      std::cout << std::setprecision(3) << std::setw(14) << (t1/t2) << ",";
      dump(ss, fos, std::cout);
    }
      std::cout << " " << std::endl;
  }
}



