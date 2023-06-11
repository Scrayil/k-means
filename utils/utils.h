// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

//
// Created by scrayil on 10/06/23.
//

#ifndef K_MEANS_UTILS_H
#define K_MEANS_UTILS_H

#include <json.hpp>

nlohmann::json parse_configuration(const std::filesystem::path& project_folder);
std::filesystem::path find_project_path();

#endif //K_MEANS_UTILS_H
