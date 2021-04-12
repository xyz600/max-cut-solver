extern crate rand;
extern crate rustc_serialize;

use rand::distributions::{Distribution, Uniform};
use rand::rngs::ThreadRng;
use rand::Rng;
use rustc_serialize::json;
use std::env;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

struct Graph {
    edges: Vec<Vec<(usize, i64)>>,
    n: usize,
}

impl Graph {
    fn new(n: usize) -> Self {
        Graph {
            edges: vec![Vec::new(); n],
            n: n,
        }
    }

    fn connect(&mut self, from: usize, to: usize, cost: i64) {
        self.edges[from].push((to, cost));
        self.edges[to].push((from, cost));
    }
}

fn load_problem(path: &str) -> Graph {
    let text = fs::read_to_string(path).unwrap();
    let content: Vec<&str> = text
        .split('\n')
        .filter(|line| line.trim().len() > 0)
        .collect();

    let token: Vec<&str> = content[0].split_whitespace().collect();
    let node_size: usize = token[0].parse().unwrap();
    let edge_size: usize = token[1].parse().unwrap();

    assert!(edge_size + 1usize == content.len());

    let mut graph = Graph::new(node_size);

    for line in &content[1..] {
        let token: Vec<&str> = line.split_whitespace().collect();

        let from: usize = token[0].parse().unwrap();
        let to: usize = token[1].parse().unwrap();
        let cost: i64 = token[2].parse().unwrap();

        graph.connect(from - 1usize, to - 1usize, cost);
    }
    graph
}

#[test]
fn load_data() {
    let data = "data/G1";
    let graph = load_problem(data);
    assert!(graph.n == 800);
}

fn calculate_energy(graph: &Graph, solution: &Vec<i64>) -> i64 {
    let mut energy = 0;
    for i in 0..graph.n {
        for (j, w) in &graph.edges[i] {
            if i < *j {
                energy += w * (1 - solution[i] * solution[*j]) / 2;
            }
        }
    }
    energy
}

fn calculate_energy_diff(graph: &Graph, solution: &Vec<i64>, pos: usize) -> i64 {
    solution[pos]
        * graph.edges[pos]
            .iter()
            .map(|(j, w)| *w * solution[*j])
            .sum::<i64>()
}

#[test]
fn energy_diff_test() {
    let data = "data/G1";
    let graph = load_problem(data);

    let mut solution = vec![0i64; graph.n];
    for i in 0..graph.n {
        solution[i] = if i % 2 == 0 { 1 } else { -1 };
    }

    let mut rng = rand::thread_rng();
    let uniform_rand = Uniform::from(0..graph.n);

    for _ in 0..10000 {
        let pos = uniform_rand.sample(&mut rng);
        let energy_before = calculate_energy(&graph, &solution);
        let energy_diff = calculate_energy_diff(&graph, &solution, pos);
        solution[pos] *= -1;
        let energy_after = calculate_energy(&graph, &solution);
        assert_eq!(energy_before + energy_diff, energy_after);
    }
}

fn accept(energy_diff: i64, progress: f64, rng: &mut ThreadRng) -> bool {
    if energy_diff >= 0 {
        true
    } else {
        let prob = (energy_diff as f64 * progress * 1e-4).exp();
        rng.gen::<f64>() <= prob
    }
}

fn simulated_annealing(graph: &Graph, timeout: u128) -> (i64, Vec<i64>) {
    let start_time = Instant::now();

    // initial solution
    let mut solution = vec![0i64; graph.n];
    for i in 0..graph.n {
        solution[i] = if i % 2 == 0 { 1 } else { -1 };
    }

    let mut energy = calculate_energy(&graph, &solution);

    let mut best_energy = energy;
    let mut best_solution = solution.clone();

    let mut counter = 0u64;
    let check_frequency = 8192;
    let mut progress = 0f64;

    let mut rng = rand::thread_rng();
    let uniform_rand = Uniform::from(0..graph.n);

    loop {
        // change state
        let pos = uniform_rand.sample(&mut rng);
        let energy_diff = calculate_energy_diff(&graph, &solution, pos);
        if accept(energy_diff, progress, &mut rng) {
            solution[pos] *= -1;
            energy += energy_diff;
        }

        if best_energy < energy {
            best_energy = energy;
            best_solution = solution.clone();
        }

        counter += 1;
        if counter % check_frequency == 0 {
            let end = start_time.elapsed().as_millis();
            if end > timeout {
                break;
            }
            progress = (end as f64) / (timeout as f64);
        }
    }
    println!("{}", counter);
    (best_energy, best_solution)
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

const INVALIDE_VALUE: usize = usize::MAX;

struct IntSet {
    data: Vec<usize>,
    index_of: Vec<usize>,
    size: usize,
}

impl IntSet {
    fn new(n: usize) -> IntSet {
        let mut ret = IntSet {
            data: vec![0; n],
            index_of: vec![0; n],
            size: n,
        };
        for i in 0..n {
            ret.data[i] = i;
            ret.index_of[i] = i;
        }
        ret
    }

    fn select(&self, rng: &mut ThreadRng) -> Option<usize> {
        if self.size > 0 {
            let uniform = Uniform::from(0..self.size);
            let pos = rng.sample(uniform);
            Some(self.data[pos])
        } else {
            None
        }
    }

    fn push(&mut self, id: usize) {
        if self.index_of[id] == INVALIDE_VALUE {
            self.data[self.size] = id;
            self.index_of[id] = self.size;
            self.size += 1;
        }
    }

    fn erase(&mut self, target: usize) {
        assert!(self.index_of[target] != INVALIDE_VALUE);
        let target_index = self.index_of[target];
        let last_index = self.size - 1;
        if target_index != last_index {
            // 消去対象の要素の場所に最終要素を上書き
            let last_value = self.data[last_index];
            self.data[target_index] = self.data[last_index];
            self.data[last_index] = INVALIDE_VALUE;
            self.index_of[target] = INVALIDE_VALUE;
            self.index_of[last_value] = target_index;
        }
        self.size -= 1;
    }

    fn len(&self) -> usize {
        self.size
    }
}

#[test]
fn test_intset() {
    let size = 128;
    let mut intset = IntSet::new(size);
    assert_eq!(intset.len(), size);
    intset.erase(10);
    assert_eq!(intset.index_of[10], INVALIDE_VALUE);
    assert_eq!(intset.len(), size - 1);

    intset.push(1);
    assert_eq!(intset.len(), size - 1);
    intset.push(10);
    assert_eq!(intset.len(), size);
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

                let (best_energy, best_solution) =
                    simulated_annealing(&graph, instance.timeout as u128);
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
