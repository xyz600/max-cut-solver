extern crate rand;
extern crate rustc_serialize;

use max_cut_solver::solver::{calculate_energy, fast_swap_greedy, load_problem};
use rustc_serialize::json;
use std::env;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};

#[derive(Debug, RustcDecodable)]
struct InstanceSettings {
    name: String,
    timeout: u64,
}

#[derive(Debug, RustcDecodable)]
struct ExperimentalSettings {
    rootpath: String,
    instance_list: Vec<InstanceSettings>,
}

struct TableWriter {
    table: Vec<Vec<String>>,
}

impl TableWriter {
    fn new(header: Vec<String>) -> TableWriter {
        let mut ret = TableWriter {
            table: vec![header],
        };
        ret.table[0].insert(0, String::new());
        ret
    }

    fn append_row(&mut self) {
        self.table.push(vec![String::new(); self.table[0].len()]);
    }

    fn save(&self, path: &str) {
        let to_oneline = |row: &Vec<String>| "|".to_string() + row.join("|").as_str() + "|";

        let write_line = |row: &Vec<String>, writer: &mut BufWriter<File>| {
            if let Err(error) = writer.write((to_oneline(row) + "\n").as_bytes()) {
                println!("error while save: {}", error);
                panic!();
            }
        };

        match File::create(path) {
            Ok(file) => {
                let mut writer = BufWriter::new(file);
                write_line(&self.table[0], &mut writer);

                let delimiter = vec!["----".to_string(); self.table[0].len()];
                write_line(&delimiter, &mut writer);

                for row in self.table.iter().skip(1) {
                    write_line(&row, &mut writer);
                }
            }
            Err(error) => {
                println!("{}", error);
            }
        }
    }
}

fn main() {
    // max-cut
    // \sum_ij G_ij (1 - x_i x_j)
    // x_i \in {-1, 1}

    let args: Vec<String> = env::args().collect();
    let config_path = &args[1];

    let maybe_settings =
        json::decode::<ExperimentalSettings>(fs::read_to_string(config_path).unwrap().as_str());

    let mut writer = TableWriter::new(vec!["timeout[ms]".to_string(), "score".to_string()]);

    match maybe_settings {
        Ok(settings) => {
            let rootpath = &settings.rootpath;
            let instance_list = &settings.instance_list;

            for instance in instance_list {
                let datapath = rootpath.clone() + instance.name.as_str();
                let graph = load_problem(datapath.as_str());

                // initial solution
                let mut initial_solution = vec![0i64; graph.n];
                for i in 0..graph.n {
                    initial_solution[i] = if i % 2 == 0 { 1 } else { -1 };
                }

                // let (best_energy, best_solution) =
                //     simulated_annealing(&graph, &mut initial_solution, instance.timeout as u128);
                let (best_energy, best_solution) =
                    fast_swap_greedy(&graph, &mut initial_solution, instance.timeout as u128);
                println!("instance name: {}", instance.name);
                println!("energy = {}", best_energy);
                // println!("solution = {:?}", best_solution);
                let last_energy = calculate_energy(&graph, &best_solution);
                println!("energy for check = {}", last_energy);
                writer.append_row();
                writer.table.last_mut().unwrap()[0] = instance.name.clone();
                writer.table.last_mut().unwrap()[1] = instance.timeout.to_string();
                writer.table.last_mut().unwrap()[2] = best_energy.to_string();
            }
            writer.save("result.md");
        }
        Err(error) => {
            println!("{}", error);
        }
    }
}
