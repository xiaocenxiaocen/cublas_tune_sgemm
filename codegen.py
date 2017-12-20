# file: codegen.py
# author: 
# date: 2017.10.19
import numpy as np
from os import system
from sys import argv
from subprocess import Popen, call, PIPE

def extractCompileInfo(kernelName):
	makeProcess = Popen("make", shell = True, stdout = PIPE, stderr = PIPE)
	makeRetStdout = makeProcess.stdout.readlines()
	makeRetStderr = makeProcess.stderr.readlines()
	import re
	spillIdx = 0
	memIdx = 0
	for idx, str_ in enumerate(makeRetStderr):
		str_ = str(str_)
		if str_.find("Function properties") != -1 and str_.find(kernelName) != -1:
			spillIdx = idx + 1
			memIdx = idx + 2
			break
	spillInfo = str(makeRetStderr[spillIdx])
	memInfo = str(makeRetStderr[memIdx])
	stack_frame = int(spillInfo[2:spillInfo.find("bytes stack frame")])
	spill_stores = spillInfo[:spillInfo.find("bytes spill stores")]
	spill_stores = int(spill_stores[spill_stores.rfind(",") + 1:])
	spill_loads = spillInfo[:spillInfo.find("bytes spill loads")]
	spill_loads = int(spill_loads[spill_loads.rfind(",") + 1:])
	
	registers = memInfo[:memInfo.find("registers")]
	registers = int(registers[registers.find("Used") + 4:])
	smem = memInfo[:memInfo.find("bytes smem")]		
	smem = smem[smem.rfind(",") + 1:]
	smem = smem[:smem.find("+")]
	smem = int(smem)
	
	ret = {"stack_frame":stack_frame, "spill_stores":spill_stores, "spill_loads":spill_loads,"registers":registers,"smem":smem}
	return ret

def gridSearch(tx, ty, ax, bx, M, K, N):
	TX = tx
	TY = ty

	order = 'COL_MAJOR'
	result = open('config_tx_%02d_ty_%02d_%s.csv'%(TX, TY, order), 'w')
	result.write("M, K, N, TX, TY, BM, BK, BN, AX, AY, BX, BY, GFLOPS\n")
	result.flush()

	opt_TX = 0
	opt_TY = 0
	opt_BM = 0
	opt_BK = 0
	opt_BN = 0
	opt_gflops = 0

	kernelName = "mysgemm_cache_AB_prefetching"

	maxrregcount = 255
	maxRegsPerBlock = 65536

	AX = ax
	BX = bx
	AY = TX * TY / AX
	BY = TX * TY / BX
	print("starting grid search...")
	for M_factor in range(2, 5):
		for K_factor in range(1, 5):
			for N_factor in range(2, 6):
	
				if order == 'ROW_MAJOR':
					BM = max(AY, TY) * M_factor
					BK = max(AX, BY) * K_factor
					BN = max(TX, BX) * N_factor	
				elif order == 'COL_MAJOR':	
					BM = max(AX, TX) * M_factor
					BK = max(AY, BX) * K_factor
					BN = max(TY, BY) * N_factor
	
				C_outputs = (BM * BN) / (TX * TY)
	
				print("test for search parameters: TX = %d, TY = %d, BM = %d, BK = %d, BN = %d, AX = %d, AY = %d, BX = %d, BY = %d"%(TX, TY, BM, BK, BN, AX, AY, BX, BY))

				cmd = '''sed "s/THREAD_BLOCK_X/%d/g;
					s/THREAD_BLOCK_Y/%d/g;
					s/ROW_BLOCK_A/%d/g;
					s/ROW_BLOCK_B/%d/g;
					s/COL_BLOCK_C/%d/g;
					s/DIM_XA/%d/g;
					s/DIM_XB/%d/g" mysgemm_template.cu > mysgemm.cu'''%(TX, TY, BM, BK, BN, AX, BX)
			
#				print cmd, "\nreplace the parameters, generate code with template"
				print("replace the parameters, generate code with template")
				call(cmd, shell = True)
				print("compile and extract compile information")
				compileInfo = extractCompileInfo(kernelName)
				spill_stores = compileInfo["spill_stores"]
				spill_loads = compileInfo["spill_loads"]
				registers = compileInfo["registers"]
				smem = compileInfo["smem"]
				print("spill stores: %d bytes, spill loads: %d bytes, registers: %d, smem: %d bytes"%(spill_stores, spill_loads, registers, smem))
				print("outputs per block: %d"%(C_outputs))
				smem = smem / 1024.0
				if smem > 48:
					continue
				if spill_loads > 1 or spill_stores > 1:
					continue
				if registers > maxrregcount:
					continue
				if registers * TX * TY > maxRegsPerBlock:
					continue

				ret = Popen("./mysgemm %d %d %d"%(M, K, N), shell = True, stdout = PIPE, bufsize = 1).stdout.readlines()
				gflopsinfo = ret[0]
				validinfo = ret[1]
				gflopsinfo = str(gflopsinfo)
				validinfo = str(validinfo)	
				gflops = float(gflopsinfo[gflopsinfo.find("=") + 2:gflopsinfo.find("GFLOPS") - 1])
				valid = validinfo[validinfo.find("=") + 2:validinfo.find("\n") - 2]

				print(valid)
				if valid == "PASS":
					print("Performance: %.2f GFLOPS"%(gflops))
					result.write("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.2f\n"%(M, K, N, TX, TY, BM, BK, BN, AX, AY, BX, BY, gflops))
					result.flush()
				
					if gflops > opt_gflops:
						opt_TX = TX
						opt_TY = TY
						opt_BM = BM
						opt_BK = BK
						opt_BN = BN
						opt_gflops = gflops
					
	cmd = '''sed "s/THREAD_BLOCK_X/%d/g;
		s/THREAD_BLOCK_Y/%d/g;
		s/ROW_BLOCK_A/%d/g;
		s/ROW_BLOCK_B/%d/g;
		s/COL_BLOCK_C/%d/g;
		s://#define VERBOSITY:#define VERBOSITY:g;
		s/DIM_XA/%d/g;
		s/DIM_XB/%d/g" mysgemm_template.cu > mysgemm.cu'''%(opt_TX, opt_TY, opt_BM, opt_BK, opt_BN, AX, BX)
	
	call(cmd, shell = True)
	Popen("make", shell = True, stdout = PIPE, stderr = PIPE).wait()

if __name__ == "__main__":

	TX = 16
	TY = 16

	AX = 32
	BX = 8

	M = int(argv[1])
	K = int(argv[2])
	N = int(argv[3])

	gridSearch(TX, TY, AX, BX, M, K, N)


