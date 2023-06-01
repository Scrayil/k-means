// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include <filesystem>
#include "k_means/k_means.h"

void sequential_version(const std::filesystem::path& project_path) {
    k_means(project_path);
}
