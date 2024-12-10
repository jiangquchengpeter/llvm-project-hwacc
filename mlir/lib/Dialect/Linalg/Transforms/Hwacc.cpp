//===- Loops.cpp - conversion from Linalg named and generic ops to loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLINALGTOHWACCPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;


#define __POSITION__ "[" << __FILE__ << ":" << __LINE__ << "] "

class SingletonLogger {
public:
  static SingletonLogger& getInstance() {
    static SingletonLogger instance;
    return instance;
  }
  template <typename T>
  SingletonLogger& operator<<(const T& data) {
    logFile << data;
    return *this; 
  }

  // Forbit the following methods
  SingletonLogger(const SingletonLogger&) = delete;
  SingletonLogger& operator=(const SingletonLogger&) = delete;

private:
    // Ensure only Construct / Deconstruct from `getInstance()`
    SingletonLogger() {
        logFile.open("debug.log", std::ios::app);  // file append
    }
    ~SingletonLogger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }

    std::ofstream logFile; // the real core
};

// static SmallVector<Value> makeCanonicalAffineApplies(OpBuilder &b, Location loc,
//                                                      AffineMap map,
//                                                      ArrayRef<Value> vals) {
//   if (map.isEmpty())
//     return {};
//   
//   assert(map.getNumInputs() == vals.size());
//   SmallVector<Value> res;
//   res.reserve(map.getNumResults());
//   auto dims = map.getNumDims();
//   for (auto e : map.getResults()) {
//     auto exprMap = AffineMap::get(dims, map.getNumSymbols(), e);
//     SmallVector<Value> operands(vals);
//     affine::canonicalizeMapAndOperands(&exprMap, &operands);
//     res.push_back(b.create<affine::AffineApplyOp>(loc, exprMap, operands));
//   }
//   return res;
// }

// template <typename LoadOpTy, typename StoreOpTy, typename OpType>
// static void inlineRegionAndEmitStore(OpBuilder &b, Location loc, OpType op,
//                                      ArrayRef<Value> indexedValues,
//                                      ArrayRef<SmallVector<Value>> indexing,
//                                      ArrayRef<Value> outputBuffers) {
//   auto &block = op->getRegion(0).front();
//   IRMapping map;
//   map.map(block.getArguments(), indexedValues);
//   for (auto &op : block.without_terminator()) {
//     auto *newOp = b.clone(op, map);
//     map.map(op.getResults(), newOp->getResults());
//   }
//   
//   Operation *terminator = block.getTerminator();
//   for (OpOperand &operand : terminator->getOpOperands()) {
//     Value toStore = map.lookupOrDefault(operand.get());
//     b.create<StoreOpTy>(loc, toStore, outputBuffers[operand.getOperandNumber()],
//                         indexing[operand.getOperandNumber()]);
//   }
// }


// /// Emits the MLIR for the scalar part of the generic op by:
// ///   1. Emitting load ops for each input and output view in order. This is
// ///      achieved by applying the appropriate input or output map to the
// ///      enclosing induction variables.
// ///   2. Emitting a call to `op.fun()` that takes as arguments the scalars
// ///      from point 1. above.
// ///   3. Emitting store ops to store the results of 2. to the output
// ///      views.
// ///
// /// An example output may resemble:
// ///
// /// ```
// ///    scf.for %i = %c0 to %0 step %c1 {
// ///      scf.for %j = %c0 to %1 step %c1 {
// ///        scf.for %k = %c0 to %4 step %c1 {
// ///          %11 = load %arg0[%i, %j] :
// ///            memref<?x?xf32, stride_specification>
// ///          %12 = load %arg1[%i, %j, %k] :
// ///            memref<?x?x?xf32, stride_specification>
// ///          %13 = load %arg2[%i, %k, %j] :
// ///            memref<?x?x?xf32, stride_specification>
// ///          %14:2 = call @foo(%11, %12, %13) : (f32, f32, f32) -> (f32, f32)
// ///          store %14#0, %arg1[%i, %j, %k] :
// ///            memref<?x?x?Xf32, stride_specification>
// ///          store %14#1, %arg2[%i, %k, %j] :
// ///            memref<?x?x?Xf32, stride_specification>
// ///       }
// ///      }
// ///    }
// /// ```
// template <typename LoadOpTy, typename StoreOpTy>
// static void emitScalarImplementation(OpBuilder &b, Location loc,
//                                      ArrayRef<Value> allIvs,
//                                      LinalgOp linalgOp) {
//   assert(linalgOp.hasPureBufferSemantics() &&
//          "expected linalg op with buffer semantics");
//   SmallVector<Value> indexedValues;
//   indexedValues.reserve(linalgOp->getNumOperands());
//   
//   auto allIvsPlusDims = SmallVector<Value>(allIvs);
//   
//   // TODO: Avoid the loads if the corresponding argument of the
//   // region has no uses.
//   // 1.a. Emit load from input operand or for scalars access the operand itself.
//   for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
//     if (linalgOp.isScalar(inputOperand)) {
//       indexedValues.push_back(inputOperand->get());
//       continue;
//     }
//     auto indexing = makeCanonicalAffineApplies(
//         b, loc, linalgOp.getMatchingIndexingMap(inputOperand), allIvsPlusDims);
//     indexedValues.push_back(
//         b.create<LoadOpTy>(loc, inputOperand->get(), indexing));
//   }
//   // 1.b. Emit load from output views.
//   for (OpOperand &outputOperand : linalgOp.getDpsInitsMutable()) {
//     SmallVector<Value> indexing = makeCanonicalAffineApplies(
//         b, loc, linalgOp.getMatchingIndexingMap(&outputOperand),
//         allIvsPlusDims);
//     indexedValues.push_back(
//         b.create<LoadOpTy>(loc, outputOperand.get(), indexing));
//   }
//   
//   // TODO: When a region inliner exists, use it.
//   // 2. Inline region, currently only works for a single basic block.
//   // 3. Emit store.
//   SmallVector<SmallVector<Value>, 8> indexing;
//   SmallVector<Value> outputBuffers;
//   for (OpOperand &outputOperand : linalgOp.getDpsInitsMutable()) {
//     if (!isa<MemRefType>(outputOperand.get().getType()))
//       continue;
//     indexing.push_back(makeCanonicalAffineApplies(
//         b, loc, linalgOp.getMatchingIndexingMap(&outputOperand),
//         allIvsPlusDims));
//     outputBuffers.push_back(outputOperand.get());
//   }
//   inlineRegionAndEmitStore<LoadOpTy, StoreOpTy>(b, loc, linalgOp, indexedValues,
//                                                 indexing, outputBuffers);
// }

// /// Replace the index operations in the body of the loop nest by the matching
// /// induction variables.
// static void replaceIndexOpsByInductionVariables(RewriterBase &rewriter,
//                                                 LinalgOp linalgOp,
//                                                 ArrayRef<Operation *> loopOps) {
//   // Extract the induction variables of the loop nest from outer to inner.
//   SmallVector<Value> allIvs;
//   for (Operation *loopOp : loopOps) {
//     llvm::TypeSwitch<Operation *>( loopOp )
//         .Case([&](scf::ForOp forOp) {
//           allIvs.push_back(forOp.getInductionVar());
//         })
//         .Default([&](Operation *op) { assert(false && "unexpected op"); });
//   }
//   assert(linalgOp.getNumLoops() == allIvs.size() &&
//          "expected the number of loops and induction variables to match");
//   // Replace the index operations in the body of the innermost loop op.
//   if (!loopOps.empty()) {
//     auto loopOp = cast<LoopLikeOpInterface>(loopOps.back());
//     for (Region *r : loopOp.getLoopRegions())
//       for (IndexOp indexOp : llvm::make_early_inc_range(r->getOps<IndexOp>()))
//         rewriter.replaceOp(indexOp, allIvs[indexOp.getDim()]);
//   }
// }


// static FailureOr<LinalgLoops> linalgOpToLoopsImpl(RewriterBase &rewriter, LinalgOp linalgOp) {
//   using LoadOpTy = memref::LoadOp;
//   using StoreOpTy = memref::StoreOp;
//   // The flattened loopToOperandRangesMaps is expected to be an invertible
//   // permutation map (which is asserted in the inverse calculation).
//   assert(linalgOp.hasBufferSemantics() && "expected linalg op with buffer semantics");
//   auto loopRanges = linalgOp.createLoopRanges(rewriter, linalgOp.getLoc());
//   auto iteratorTypes = linalgOp.getIteratorTypesArray();
//   SmallVector<Value> allIvs;
//   GenerateLoopNest<scf::ForOp>::doit(
//       rewriter, linalgOp.getLoc(), loopRanges, linalgOp, iteratorTypes,
//       [&](OpBuilder &b, Location loc, ValueRange ivs, ValueRange operandValuesToUse) -> scf::ValueVector {
//         assert(operandValuesToUse == linalgOp->getOperands() && "expect operands are captured and not passed by loop argument");
//         allIvs.append(ivs.begin(), ivs.end());
//         emitScalarImplementation<LoadOpTy, StoreOpTy>(b, loc, allIvs, linalgOp);
//         return scf::ValueVector{};
//       });
//   // Number of loop ops might be different from the number of ivs since some
//   // loops like affine.parallel and scf.parallel have multiple ivs.
//   SetVector<Operation *> loopSet;
//   for (Value iv : allIvs) {
//     if (!iv) return failure();
//     // The induction variable is a block argument of the entry block of the
//     // loop operation.
//     BlockArgument ivVal = dyn_cast<BlockArgument>(iv);
//     if (!ivVal) return failure();
//     loopSet.insert(ivVal.getOwner()->getParentOp());
//   }
//   LinalgLoops loops(loopSet.begin(), loopSet.end());
//   // Replace all index operations in the loop body.
//   replaceIndexOpsByInductionVariables(rewriter, linalgOp, loops);
//   return loops;
// }




















LogicalResult appendMangledType(llvm::raw_string_ostream &ss, Type t) {
  if (auto memref = llvm::dyn_cast<MemRefType>(t)) {
    ss << "view";
    for (auto size : memref.getShape())
      if (size < 0)
        ss << "sx";
      else
        ss << size << "x";
    if (failed(appendMangledType(ss, memref.getElementType())))
      return failure();
    if (auto as = memref.getMemorySpace()) {
      if (auto attr = llvm::dyn_cast<IntegerAttr>(as))
        ss << "as" << attr.getInt();
      else
        return failure();
    }
    return success();
  }
  if (auto vec = llvm::dyn_cast<VectorType>(t)) {
    ss << "vector";
    llvm::interleave(
        vec.getShape(), [&](int64_t i) { ss << i; }, [&]() { ss << "x"; });
    if (failed(appendMangledType(ss, vec.getElementType())))
      return failure();
    return success();
  } else if (t.isSignlessIntOrIndexOrFloat()) {
    ss << t;
    return success();
  }
  return failure();
}


std::string MatmulOp_generateLibraryCallName(Operation *op) {
  assert(isa<LinalgOp>(op));
  std::string name(op->getName().getStringRef().str());
  std::string fun = "";
  for (NamedAttribute kv : op->getAttrs()) {
    if (UnaryFnAttr ufa = llvm::dyn_cast<UnaryFnAttr>(kv.getValue())) {
      fun = stringifyEnum(ufa.getValue()).str() + "_";
    } else if (BinaryFnAttr bfa = llvm::dyn_cast<BinaryFnAttr>(kv.getValue())) {
      fun = stringifyEnum(bfa.getValue()).str() + "_";
    }
  }
  name.reserve(128);
  std::replace(name.begin(), name.end(), '.', '_');
  llvm::raw_string_ostream ss(name);
  ss << "_" << fun;
  for (Type t : op->getOperandTypes()) {
    if (failed(appendMangledType(ss, t)))
      return std::string();
    ss << "_";
  }
  std::string res = ss.str();
  res.pop_back();
  return res;
}



static MemRefType makeStridedLayoutDynamic(MemRefType type) {
  return MemRefType::Builder(type).setLayout(StridedLayoutAttr::get(
      type.getContext(), ShapedType::kDynamic,
      SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic)));
}

/// Helper function to extract the operand types that are passed to the
/// generated CallOp. MemRefTypes have their layout canonicalized since the
/// information is not used in signature generation.
/// Note that static size information is not modified.
static SmallVector<Type, 4> extractOperandTypes(Operation *op) {
  SmallVector<Type, 4> result;
  result.reserve(op->getNumOperands());
  for (auto type : op->getOperandTypes()) 
  {
    // The underlying descriptor type (e.g. LLVM) does not have layout information. 
    // Canonicalizing the type at the level of std when going into a library call avoids needing to introduce DialectCastOp.
    if (auto memrefType = dyn_cast<MemRefType>(type)) 
      result.push_back(makeStridedLayoutDynamic(memrefType));
    else 
      result.push_back(type);
  }
  return result;
}

static FailureOr<FlatSymbolRefAttr> getLibraryCallSymbolRef(Operation *op, PatternRewriter &rewriter) {
  assert(isa<MatmulOp>(op) && "expected a matmul op here");
  auto matmulOp = cast<MatmulOp>(op);
  // auto fnName = MatmulOp_generateLibraryCallName(matmulOp);
  auto fnName = matmulOp.getLibraryCallName();
  FlatSymbolRefAttr fnNameAttr = SymbolRefAttr::get(rewriter.getContext(), fnName);  // fnName is a dynamic std::string, unique it via a SymbolRefAttr.

  SingletonLogger::getInstance() << __POSITION__ <<"fnNameAttr: Attr=" << fnNameAttr.getAttr().str() << " Value=" << fnNameAttr.getValue().str() << "\n";

  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnNameAttr.getAttr())) return fnNameAttr;

  SingletonLogger::getInstance() << __POSITION__ <<"module: Name=" << module->getName().getStringRef().str() << "\n";


  SmallVector<Type, 4> inputTypes(extractOperandTypes(op));
  if (op->getNumResults() != 0) return rewriter.notifyMatchFailure( op, "Library call for linalg operation can be generated only for ops that have void return types");

  SingletonLogger::getInstance() << __POSITION__ <<"op: InputCnt=" << matmulOp->getNumOperands() << " OutputCnt=" << matmulOp->getNumResults() << "\n";

  auto libFnType = rewriter.getFunctionType(inputTypes, {});

  OpBuilder::InsertionGuard guard(rewriter);

  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  func::FuncOp funcOp = rewriter.create<func::FuncOp>(op->getLoc(), fnNameAttr.getValue(), libFnType);
  // Insert a function attribute that will trigger the emission of the
  // corresponding `_mlir_ciface_xxx` interface so that external libraries see
  // a normalized ABI. This interface is added during std to llvm conversion.
  funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                  UnitAttr::get(op->getContext()));
  funcOp.setPrivate();

  return fnNameAttr;
}

static SmallVector<Value, 4> createTypeCanonicalizedMemRefOperands(OpBuilder &b, Location loc, ValueRange operands) {
  SmallVector<Value, 4> res;
  res.reserve(operands.size());
  for (auto op : operands) {
    auto memrefType = dyn_cast<MemRefType>(op.getType());
    if (!memrefType) { res.push_back(op); continue; }
    Value cast = b.create<memref::CastOp>(loc, makeStridedLayoutDynamic(memrefType), op);
    res.push_back(cast);
  }
  return res;
}













namespace {

class LinalgRewritePattern : public RewritePattern {
public:
  LinalgRewritePattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // SingletonLogger::getInstance() << "Test:[" << op->getName().getStringRef().str() << "]\t";

    // auto linalgOp = cast<LinalgOp>(op);
    // auto fnName = linalgOp.getLibraryCallName();

    auto matmulOp = dyn_cast<MatmulOp>(op);
    if (!isa<MatmulOp>(op) || !matmulOp.hasPureBufferSemantics()) {
      // SingletonLogger::getInstance() << "Fail To Match!" << "\n";
      return rewriter.notifyMatchFailure(
          op, "expected MatmulOp with buffer semantics");
    }

    SingletonLogger::getInstance() << __POSITION__ <<"[[" << matmulOp->getName().getStringRef().str() << "]] \t";
    SingletonLogger::getInstance() << __POSITION__ <<"getLibraryCallName: " << matmulOp.getLibraryCallName() << "\n";

    // if (fnName.empty()) return rewriter.notifyMatchFailure(op, "No library call defined for: ");

    auto libraryCallName = getLibraryCallSymbolRef(matmulOp, rewriter);
    if (failed(libraryCallName)) return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(
      op, 
      libraryCallName->getValue(), 
      TypeRange(), 
      createTypeCanonicalizedMemRefOperands(rewriter, op->getLoc(), op->getOperands())
    );

    // if (failed(linalgOpToLoopsImpl(rewriter, matmulOp))) return failure();
    // rewriter.eraseOp(op);

    SingletonLogger::getInstance() << "\n";

    return success();
  }
};



static void lowerLinalgToHwaccImpl(Operation *enclosingOp) {
  MLIRContext *context = enclosingOp->getContext();
  RewritePatternSet patterns(context);
  patterns.add<LinalgRewritePattern>(context);
  // memref::DimOp::getCanonicalizationPatterns(patterns, context);
  // tensor::DimOp::getCanonicalizationPatterns(patterns, context);
  // affine::AffineApplyOp::getCanonicalizationPatterns(patterns, context);
  // Just apply the patterns greedily.
  (void)applyPatternsAndFoldGreedily(enclosingOp, std::move(patterns));
}


struct LowerToHwacc : public impl::ConvertLinalgToHWACCPassBase<LowerToHwacc> {
  using impl::ConvertLinalgToHWACCPassBase<
      LowerToHwacc>::ConvertLinalgToHWACCPassBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    lowerLinalgToHwaccImpl(getOperation());
  }
};


} // namespace


// std::unique_ptr<Pass> mlir::createConvertLinalgToHwaccPass() {
//   return std::make_unique<LowerToHwacc>();
// }



// /// Emits a loop nest of `scf.for` with the proper body for `linalgOp`.
// FailureOr<LinalgLoops> mlir::linalg::linalgOpToLoops(RewriterBase &rewriter,
//                                                      LinalgOp linalgOp) {
//   return linalgOpToLoopsImpl<scf::ForOp>(rewriter, linalgOp);
// }
