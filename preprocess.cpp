/**
 * @file preprocess.cpp
 * @brief Preprocess data before training, return data to python
*/

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/parquet/api/reader.h>

int main() {

    std::string filepath = "";
    std:cin >> filepath;

    // initialize parquet reader
    std::shared_ptr<arrow::io::ReadableFile> file;
    arrow::io::ReadableFile::Open(filepath, arrow::default_memory_pool(), &file);
    std::shared_ptr<parquet::arrow::FileReader> reader;
    parquet::arrow::OpenFile(file, arrow::default_memory_pool(), &reader);

    // get row group count
    int numRowGroups = reader->num_row_groups();

    // Process each row group
    for (int i = 0; i < numRowGroups; i++) {
        // Read a row group
        std::shared_ptr<arrow::Table> table;
        reader->ReadRowGroup(i, &table);

        // Process the data in the row group
        // ...

        // Access the columns in the row group
        for (int j = 0; j < table->num_columns(); j++) {
            std::shared_ptr<arrow::ChunkedArray> column = table->column(j);
            // Process the column data
            // ...
        }
    }

    return 0;
}