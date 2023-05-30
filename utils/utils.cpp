// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include <fstream>
#include <iostream>
#include <thread>

#include "../utils/utils.h"

// FUNCTIONS

/**
 * This function allows to parse and load the project's json configuration.
 *
 * All the items in the configuration are loaded into a proper structure.
 * @param project_folder This is the path related to the current project.
 * @return A json object that contains all the specified parameters.
 */
nlohmann::json parse_configuration(const std::filesystem::path& project_folder) {
    std::filesystem::path config_path = project_folder / "config" / "default.json";
    std::ifstream config_file;
    config_file.open(config_path);
    nlohmann::json json_config = nlohmann::json::parse(config_file);
    config_file.close();
    return json_config;
}

/**
 * This function allows to retrieve the current project's path in the filesystem.
 *
 * It perform a directory traversal until the current project's name folder is found.
 * The folder might be named as specified for this to work.
 * @return A path object representing the projects location.
 */
std::filesystem::path find_project_path() {
    std::filesystem::path project_folder = std::filesystem::current_path();
    while(!project_folder.string().ends_with("DESCracker"))
        if(project_folder.string() == "/") {
            project_folder.clear();
            break;
        } else {
            project_folder = project_folder.parent_path();
        }
    return project_folder;
}