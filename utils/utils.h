// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#ifndef RANDOMMAZESOLVER_UTILS_H
#define RANDOMMAZESOLVER_UTILS_H

#include <json.hpp>


nlohmann::json parse_configuration(const std::filesystem::path& project_folder);
std::filesystem::path find_project_path();

#endif //RANDOMMAZESOLVER_UTILS_H