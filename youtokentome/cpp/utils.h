#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "third_party/flat_hash_map.h"

namespace srcd {
const uint32_t SPACE_TOKEN = 9601;

typedef void (*py_write_func)(void *self, const char *buffer, int size);
typedef int (*py_read_func)(void *self, char *buffer, int size);
typedef std::string (*py_name_func)(void *self);

struct StreamWriter {
  virtual void write(const char *buffer, int size) = 0;
  virtual std::string name() const noexcept = 0;
  virtual ~StreamWriter() = default;

  static std::unique_ptr<StreamWriter> open(const std::string &file_name);
  static std::unique_ptr<StreamWriter> assemble(
      py_write_func write, py_name_func name, void *self);
};

struct StreamReader {
  virtual int read(char *buffer, int size) = 0;
  virtual std::string name() const noexcept = 0;
  virtual ~StreamReader() = default;

  std::string read_all();
  static std::unique_ptr<StreamReader> open(const std::string &file_name);
  static std::unique_ptr<StreamReader> assemble(
      py_read_func read, py_name_func name, void *self);
};

struct BPE_Rule {
  // x + y -> z
  uint32_t x{0};
  uint32_t y{0};
  uint32_t z{0};

  BPE_Rule() = default;

  BPE_Rule(uint32_t x, uint32_t y, uint32_t z);

  bool operator==(const BPE_Rule &other) const;
};

struct SpecialTokens {
  int pad_id = -1;
  int unk_id = -1;
  int bos_id = -1;
  int eos_id = -1;

  SpecialTokens() = default;

  SpecialTokens(int pad_id, int unk_id, int bos_id, int eos_id);

  void dump(StreamWriter &fout);

  void load(StreamReader &fin);

  uint32_t max_id() const;

  bool taken_id(int id) const;

  size_t n_special_tokens() const;
};

struct BpeConfig {
  double character_coverage = 1;
  int n_threads = 0;
  SpecialTokens special_tokens;

  BpeConfig() = default;

  BpeConfig(double character_coverage, int n_threads,
            const SpecialTokens &special_tokens);
};

struct Status {
  int code{0};
  std::string message;
  Status() = default;
  Status(int code, std::string message);

  const std::string &error_message() const;
  bool ok() const;
};

struct BPEState {
  ska::flat_hash_map<uint32_t, uint32_t> char2id;
  std::vector<BPE_Rule> rules;
  SpecialTokens special_tokens;

  void dump(StreamWriter &fout);

  Status load(StreamReader &fin);
};

struct DecodeResult {
  std::vector<int> ids;
  std::vector<std::string> pieces;
};

struct EncodingConfig {
  bool bos;
  bool eos;
  bool reverse;
};

bool is_space(uint32_t ch);

std::vector<std::string> read_lines_from_stdin(size_t batch_limit, size_t *processed);

template<typename T>
void write_to_stdout(const std::vector<std::vector<T>> &sentences, bool flush) {
  for (const auto &sentence : sentences) {
    for (const auto &token : sentence) {
      std::cout << token << " ";
    }
    std::cout << "\n";
  }
  if (flush) {
    std::cout << std::flush;
  }
}

}  // namespace srcd
