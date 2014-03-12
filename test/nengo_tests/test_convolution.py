import numpy as np
# import matplotlib.pyplot as plt
import math

import sys
sys.path.append(sys.argv[1])
import nef_theano as nef

import functions as funcs

# DxD discrete fourier transform matrix
def discrete_fourier_transform(D):
    m=[]
    for i in range(D):
        row=[]
        for j in range(D):
            row.append(complex_exp((-2*math.pi*1.0j/D)*(i*j)))
        m.append(row)
    return m

# DxD discrete inverse fourier transform matrix
def discrete_fourier_transform_inverse(D):
    m=[]
    for i in range(D):
        row=[]
        for j in range(D):
            row.append(complex_exp((2*math.pi*1.0j/D)*(i*j))/D)
        m.append(row)
    return m

# formula for e^z for complex z
def complex_exp(z):
    a=z.real
    b=z.imag
    return math.exp(a)*(math.cos(b)+1.0j*math.sin(b))

def output_transform(dimensions):
    ifft=np.array(discrete_fourier_transform_inverse(dimensions))

    def makeifftrow(D,i):
        if i==0 or i*2==D: return ifft[i]
        if i<=D/2: return ifft[i]+ifft[-i].real-ifft[-i].imag*1j
        return np.zeros(dimensions)
    ifftm=np.array([makeifftrow(dimensions,i) for i in range(dimensions/2+1)])

    ifftm2=[]
    for i in range(dimensions/2+1):
        ifftm2.append(ifftm[i].real)
        ifftm2.append(-ifftm[i].real)
        ifftm2.append(-ifftm[i].imag)
        ifftm2.append(-ifftm[i].imag)
    ifftm2=np.array(ifftm2)

    return ifftm2.T

def input_transform(dimensions, first, invert=False):
    fft=np.array(discrete_fourier_transform(dimensions))

    M=[]
    for i in range((dimensions/2+1)*4):
        if invert: row=fft[-(i/4)]
        else: row=fft[i/4]
        if first:
            if i%2==0:
                row2=np.array([row.real,np.zeros(dimensions)])
            else:
                row2=np.array([row.imag,np.zeros(dimensions)])
        else:
            if i%4==0 or i%4==3:
                row2=np.array([np.zeros(dimensions),row.real])
            else:
                row2=np.array([np.zeros(dimensions),row.imag])
        M.extend(row2)
    return M

def circconv(a, b):
    return np.fft.ifft(np.fft.fft(a)*np.fft.fft(b)).real

D = 32
subD = 16
N = 100
N_C = 500

net = nef.Network('Main', fixed_seed=1, command_arguments=sys.argv[2:],
  usr_module='test/nengo_tests/functions.py')

A = np.random.randn(D)
A /= np.linalg.norm(A)

# 16, 8
#A = [-0.22163856910302998, -0.16642805390773005, 0.03510604578619818, 0.005427641075043549, -0.16266567374032798, -0.005160354921985649, -0.2789527793900736, -0.23986306256060722, 0.14979044442087716, 0.28816467169038995, -0.028386540541611566, -0.2645078463985762, -0.23065830305609872, 0.1981870653448343, -0.6664351903718526, 0.21729468674612495]
# 512, 16
# A = [-0.04846746786881538, 0.010464966083395713, -0.06503481073011917, 0.0883599453782411, -0.018074903409014322, 0.054800393688337744, -0.046308863278226926, -0.030316475525555886, 0.025011942692711637, 0.07181465980348624, 0.048091143903690896, -0.005526905189011298, 0.05225672077400424, 0.01475156182781303, -0.04407333832928844, 0.004247775317004689, 0.016469477421476194, -0.06663735860973981, -0.016635938565318767, 0.06735558216089081, 0.04682678734284345, 0.001708382901852913, 0.018969988836049247, -0.0049204611152135185, 0.022806939840894595, 0.0020851341646266138, 0.029121701575284112, 0.002338741551765697, -0.027829513370027457, 0.021430156017165577, -0.0077323799480073785, 0.05097475836211078, 0.05122243686470942, 0.00013046254108120545, -0.016176173828732848, 0.011853053424000117, 0.06473193743041307, 0.011619011144138457, 0.008035483229314754, 0.06304428260854732, 0.03733396998038378, 0.001942021673804798, 0.046499690911610506, -0.02141588695164249, -0.03672389175842681, 0.03297895614337031, 0.0009953930303805455, -0.030966267918851608, -0.06096020549502633, -0.085320601607696, 0.056277426842214845, 0.006205688220724862, 0.01069544282254401, 0.02680150801810538, -0.0196315581819461, -0.03969013882478792, 0.02207841150759627, -0.03047556983103888, -0.045397634816149286, -0.05685041830378666, 0.005560271809541184, -0.02107139218512419, 0.033192175277665194, -0.02212340182003014, 0.014039036832833534, -0.06255476971807865, 0.104455156691934, -0.008986081452428818, 0.015913280528139964, -0.10330655776376693, 0.008008143128447012, -0.016388963921469397, -0.025286982887790976, -0.00026586765714485636, -0.05898351743472187, -0.08654489534409351, -0.008667886244097051, -0.00761111730052323, 0.020094816822775138, -0.03165684074804173, 0.06162949124024127, -0.039361028033385315, 0.023783589326802104, -0.04125895153652768, 0.010605925264364604, -0.03426554378249938, -0.010196949301553, 0.00988372871938729, -0.08910549715848022, -0.05114036465123451, -0.007161900538520966, -0.02908714896332318, 5.712579040398648e-05, -0.06651741983157464, -0.006816704306478765, 0.043197374968704266, -0.014860520818743799, 0.08851275507336177, 0.01970976624219633, -0.03944384522601803, -0.0417778443875154, 0.032176026172888235, -0.016580187015283562, 0.026157049266245264, 0.03562938580961668, 0.0016832981957511145, 0.014722204179466833, -0.045726070785581205, 0.05003290655038566, -0.045385976747285665, -0.023475561388550354, -0.018793968164413772, -0.0689898174222769, -0.027830429327999753, -0.015421516887568522, -0.03400354244146824, -0.06009318823675101, 0.05420391506024433, -0.02138997225011041, -0.020286584538348137, 0.06301610298379995, -0.026295744995572592, -0.034670826884584456, 0.03303483635740065, -0.003423282244774795, -0.07054722663635529, 0.033766341905504864, -0.0017584439349309362, 0.016231061134639884, 0.02197806427179765, -0.010656941307111435, 0.05583902669436475, 0.03105730610383494, 0.050198642895001803, 0.02202413235138754, -0.012374468653025486, -0.01815256752563048, 0.029783041351680584, 0.061550193158111284, 0.021390327022754087, 0.06278113409671131, -0.02590217094358433, -0.045543831395238946, -0.09895118933277461, 0.00045524467495502015, 0.03292272698500619, 0.06907069113816179, 0.0025160425271344472, 0.09084281461212901, -0.09492922851641386, 0.014594894027366781, -0.03814906020309651, 0.0125464962756018, 0.05332530033245084, 0.013860019993880647, 0.018917450932015613, 0.00746726646692159, 0.006549510972652343, 0.02031000332365612, 0.08239259260311227, 0.022258541177876116, 0.008964272389332045, -0.028384906122045944, 0.0040686867537745915, -0.05115510755758482, -0.023065686909110837, 0.0651939886878052, -0.06388675611354391, -0.025271594214312132, 0.008600927640439265, -0.032650986213334815, 0.05619188943124665, -0.06200140257209617, -0.0807026927591864, -0.0035715192859333998, 0.02947356615540654, -0.09879969423072942, 0.05110359618486923, -0.010862018223730905, -0.02313146727626875, -0.03671110788713915, -0.014934670212051495, -0.05988712201815885, 0.01685889319721745, 0.07284789001146187, 0.02680344285703014, 0.07807969765123046, -0.006602957305335958, 0.013273825588688089, -0.0176573773563561, -0.051527703785396284, 0.03079122567159637, -0.023505226175149115, -0.01561031191435961, 0.0005648499274128217, -0.03456398854395515, 0.026238916728853402, -0.09796907954706796, 0.00636434291051968, -0.03662807453165307, 0.014534197267682127, -0.026355060997783807, 0.11340041837547912, 0.016169368152104194, -0.01063429267451642, 0.01918546394305481, -0.021746592567065442, 0.03950337489694668, 0.07927243218223201, -0.035169953569435836, -0.03228114924511188, 0.0006227075769331905, -0.054380040761716264, -0.03545090666254275, -0.08929181662074724, -0.06591421719192309, 0.06295269873926182, 0.035889299482237194, -0.08163776420542367, 0.07992272798215952, 0.016991998725879223, 0.07121972610930173, 0.004441716822807892, 0.015572074641647364, 0.07744307689289162, 0.005558529412930641, 0.022490521723029577, -0.004435867685827523, 0.022106931517559256, -0.04971429527476297, -0.048815151800259526, 0.05618629722722604, 0.0036290924672263106, -0.025232840811869925, 0.03486455054853868, -0.04263424301276407, -0.028648900172023123, -0.14633862704873452, -0.002867297309340084, -0.04837149269408737, -0.0001278351371752018, 0.05351663716119046, 0.09682501153594768, -0.0931471273935264, -0.04226077491303433, 0.03616297488591874, 0.05229760388484443, -0.042114040164088845, -0.02743651746337007, -0.029398484986481413, -0.014420910605949635, -0.02790702298104135, -0.009266988410965687, 0.05211876842418107, 0.011534498837535366, -0.05411559204046603, 0.06873685983613853, -0.0716390370746217, -0.040611327570696125, -0.013623927216739244, 0.03480082076587049, 0.042261978710123105, 0.0031657769514614473, 0.020984596390779247, 0.022907908124315795, -0.02394202774346694, 0.0743903152149698, -0.03698522830822875, 0.02724028088835009, 0.00113888919236199, 0.07922326417039005, 0.02380271171865187, -0.0536449905955057, -0.04476755501428005, 0.016155531950493487, 0.01605396857826173, -0.010627459386466372, -0.026211403920598522, -0.07423929159947401, 0.025218932923528035, -0.024348456599198452, -0.04655557370905368, 0.0639013835543778, -0.06440131149942872, -0.028634275450342344, 0.029570762628490444, 0.07190576864919064, -0.018808727542385852, 0.07202027643495537, -0.07844077368343087, 0.0430418800110257, -0.013894289404714485, 0.05291691148790661, -0.005875611847456866, -0.02456626086725292, 0.09228431496424797, 0.08491903712890646, 0.024309565923949623, 0.08769464453916923, 0.010108980074040807, -0.05443812745227736, -0.04322564829769521, -0.0722406578385242, 0.04823198809465236, -0.00031038155921611394, 0.011997489789337703, 0.022788233395080123, -0.009492108022991701, -0.0103523083362404, -0.09328003214683199, -0.01298305043828431, 0.024441596835186773, 0.033550655479972914, 0.015331343092526512, -0.06430582568234887, 0.051742393282379055, 0.015478673127593746, -0.04077664808675785, -0.047699055800093024, 0.01560202717715283, 0.03671096381397526, 0.01857301416403308, -0.03867089670234192, 0.05287688631111564, -0.034341255081813786, -0.0318038335061382, -0.00025776154804617925, 0.023712167089044695, -0.0013605526473580937, 0.034419554355777635, 0.032588227124969386, 0.04324271046704723, 0.06833422192074919, 0.06683316162430708, -0.01690892823805122, 0.012775302019944378, 0.022126694188719022, 0.02477343303846815, 0.03148432248453898, -0.009675951019149849, 0.03163557943472117, 0.014405527285783612, -0.09119544690154215, 0.06544755448520787, 0.031271768175945173, -0.008580642102689556, -0.01092653944196497, -0.04290200080428372, 0.042198941716151704, -0.0805066318928938, -0.03341853950791407, -0.01243593710919566, 0.04075827665878218, -0.050050807529589555, 0.01386769489163138, 0.07052321838680349, 0.032976498399687956, -0.010499991677149162, -0.07100398727143273, 0.026624820858546845, -0.0443154722412284, -0.030861895948054455, -0.055529348768480966, 0.040640527517272915, 0.011129606190293653, -0.018845228280619783, -0.06097959007428612, -0.05383597030821173, 0.06270727861583468, 0.0481334605822405, -0.0182657527518536, -0.02531698384580953, -0.06419917135052286, 0.026295784247661467, 0.044169543343639525, 0.03116045836896428, 0.034924922929956304, 0.04837477723412656, 0.05989981810391972, 0.008433655227701295, -0.01292571847194274, -0.024372214004089107, 0.04746113775230838, 0.014885224903557972, 0.03590641740519605, -0.037640641761345704, 0.0510453699172432, -0.022690645068581577, 0.010177210855419644, 0.009985631018810835, 0.0252030515281674, 0.0017345443260606522, 0.04868157128732865, -0.028729874397334272, 0.025634562538597423, -0.0444911941312521, 0.0036782149679417315, 0.01637541007898009, 0.03016250571839622, -0.06615459644303164, 0.0301261232132403, 0.06035164821601804, 0.02483832364767019, 0.01538296586338666, 0.036209278281300834, 0.03291781947669913, -0.02735765593493547, -0.03122657779595365, -0.0014747418190553562, 0.058000094015124765, -0.022340260258284134, -0.025905899924256626, 0.04696157248187023, 0.11026187203629735, -0.031017642588396545, -0.019940246707294204, 0.05111042724758896, 0.0008417171762088041, -0.03840135539423782, 0.015660805352017755, -0.06898292305517333, -0.011757336269470212, -0.00951155803984348, 0.0022495833656535854, 0.0789476506523377, -0.08347892389427562, -0.02925708764515564, -0.09020443150103054, 0.025986187026173776, 0.09337680238903229, 0.032715449675045825, 0.015549157979477, 0.07753723253486973, 0.06323919300904343, -0.005264922310230087, -0.009874854678350725, -0.021480823486476577, 0.019680637816993782, -0.009413185869911685, 0.034744980117656636, -0.017873695888835046, -0.0028911175318567886, -0.0038881638256694846, 0.018284195547774856, 0.04980973566750357, -0.007425018539216158, 0.022467830111262875, -0.05005797586923718, 0.07975946827871375, 0.027320607720071777, 0.020527826275316376, -0.003646948620901677, -0.019947287244883243, 0.041782980302144175, 0.007113028111423351, 0.01678450524175185, 0.05301787069799068, -0.07340451297851935, 0.04036791441398689, -0.02583429885398439, -0.1121865235521737, 0.06385734204389146, -0.06971015215026688, 0.018348071302434956, 0.11864635668008001, 0.03265444439968055, 0.043173775878735755, -0.06969487208850544, -0.006546447871047623, -0.04466112905320209, 0.007468330975862345, -0.015423018917972733, 0.0660153012981035, -0.019730418122037657, 0.024082764747708197, 0.0909849804130505, 0.0426937611141851, 0.06657437217801966, 0.03971400020129477, 0.012795155043639884, -0.0008970635681659526, 0.026201321256521085, -0.023586201511274055, 0.03641527620619858, 0.043418308058758855, 0.05648101424502138, 0.0055508242358621885, -0.00904200718933734, -0.05757924864520587, -0.01398543296076121, 0.07742207836069569, 0.014471368302299666, -0.06822928158059637, -0.0421615422251272, -0.004816513879564943, 0.013225346604517595, -0.01954515865973366, -0.015095443494203912, 0.037409448605767164, -0.027060827822168843, 0.010488740325207734, -0.11611526879164465, -0.01076136371515272, 0.015184192987627137, 0.044597648902474085, -0.04785055941696709, -0.0544717433473682, -0.035914353453465206, -0.018823490411987373, -0.014929873817560089, -0.08308573727405519, 0.004515281257666244]

B = np.random.randn(D)
B /= np.linalg.norm(B)

# 16, 8
#B = [-0.0070677433391332966, -0.14798361048090658, -0.37178897344892375, -0.42978348334732663, 0.14216472529226645, -0.29494305458524744, -0.42195366674581736, -0.13157963774082326, -0.09692083031536362, 0.013566785306620442, 0.0644473264537378, 0.2297653382672786, -0.3722871311829867, -0.1084945678039673, 0.10241790709779663, -0.3537912119211055]
# 512, 16
# B = [0.04819999977314152, -0.052547990028829435, 0.055774815641962606, -0.007525680474403822, 0.06651953241873768, -0.05152416431699123, 0.022515599437827896, 0.021872567921840457, -0.00795561216696092, 0.04000058706374679, -0.044290623140753944, -0.01497819310481376, -0.044581005537384796, -0.07944180271508691, -0.05307195322658219, 0.016661291735176487, 0.059061547474597804, 0.029655574670377202, -0.005195275209565573, 0.05358615680441934, -0.051334804889635786, 0.0038999814040307436, -0.04693165195302847, 0.04791184338315176, -0.010811563263749813, -0.010413406143379421, 0.02607201924843158, 0.031161104081615397, -0.03567676539486815, -0.05033907694732027, 0.036503622576068526, -0.058806427755630726, -0.029704460956646594, 0.08643320855263466, -0.06613174718580427, -0.03380079208305996, 0.0041218082374209755, 0.019027119319592858, 0.010354282328584015, -0.03999463121289678, -0.010395104924262127, 0.08604011248365373, -0.017996149330445427, 0.031305716996538314, 0.0006710624707357756, 0.008478217398635274, 0.023811427012551213, -0.007369538846906448, -0.025600156312087934, 0.010714014647654264, -0.05062670151900221, -0.059095666893100196, -0.03554412772747951, 0.011570089974060655, 0.03001675779078502, 0.017029765704831463, 0.07768160701174626, 0.01063784085261291, 0.04871527535668551, -0.038900231599282936, -0.0013787023641839776, 0.0381206255010237, 0.08256517767242788, 0.01738558659312116, -0.04597075262056571, -0.007976169025099601, 0.0664639095022131, -0.05308306080329494, -0.02463156929467683, 0.06435962526266621, -0.038390134660077284, 0.012222661896313007, -0.07171683989188159, 0.06928165228962155, -0.09296978328316313, -0.018843972637423195, -0.0642338096879752, 0.0010394198527230822, -0.04011174593716194, -0.019250010442370753, -0.053740076236409605, 0.01270361088629603, -0.012613377495743223, -0.036448703497456617, 0.05720251428555438, 0.026851969619987103, -0.05156700466835821, 0.05457869822532907, -0.018034839510394116, -0.02423259540802232, 0.0008044697810106103, 0.09304241472463798, -0.041210732486987714, 0.09309803173647709, -0.05587841956287623, 0.06670906093650548, -0.005696800667221294, 0.015656973748062332, -0.03499110998030779, 0.09216329122951526, -0.00648931639219505, -0.010398845751725179, -0.032486301079309304, 0.03704263215345774, -0.060616930900538164, 0.017916035176888848, 0.04375124558452518, 0.04913317446010148, 0.11555271839896834, -0.041473796638413454, -0.08492917303887318, 0.002984810574527221, -0.010378853232771925, 0.04263260371290989, -0.03935201568071246, -0.0250790176060448, 0.0735262481827474, 0.028189475736286385, 0.046999507388414546, 0.05246469655816062, 0.05105960619140015, -0.053909500386135505, -0.016200994111276627, -0.022967332524172303, -0.060317810030572294, -0.004694894083903053, -0.006643387643159883, 0.09782893148022731, 0.025799797361859108, 0.01376224639732659, -0.002718759644576821, 0.03655371470424807, -0.054979462173507326, 0.05452750485363238, -0.042398071650898636, 0.044001546017369775, -0.003421870174894503, 0.04985062920732902, 0.048050154665269254, -0.013199344528150523, 0.11824350617172649, 0.008158840198540013, -0.023117596120068803, 0.023213766542975123, 0.02867628989745894, -0.017845428976462126, 0.010290542544884922, -0.00013410250281366336, 0.01769620421364338, 0.016700506655265936, 0.02339752162526471, 0.03709883427457516, 0.024623914582038317, 0.04077352443481282, 0.06384703388809812, -0.016175800788719625, 0.01000760658784601, -0.05073942016588056, 0.0021796543749569267, 0.00809741318948683, -0.11170481158263486, 0.048938368725452645, -0.02006286803989284, -0.077596766679995, 0.07632266195686153, -0.019930913064193433, 0.02753315685962733, -0.0065327859828665695, -0.016889670551704598, -0.020621673278269744, -0.12181664639926175, 0.054916545338357284, 0.023864243187904782, 0.024820020727680286, 0.0021204064906330264, 0.04393658225592372, -0.051028444524400245, -0.07419818965167504, -0.04279160134126293, 0.03545671667504893, -0.10619644561851058, -0.04084131928055951, -0.028837034255359495, 0.037869619091363334, -0.008123223764236095, -0.03564745676618344, -0.02493722737986169, -0.006797587810196462, 0.038121571170172966, 0.03462024274861439, -0.06805373771139854, -0.039613213596427915, -0.01235903924386059, 0.0032141841534370605, -0.04159766296873112, 0.06937070395420557, -0.008752453831098713, -0.04881576896296146, 0.012126615631241126, 0.016497061703446506, -0.009948815214450879, -0.02909900987091385, 0.042290394155050176, 0.060363010221386945, -0.012330091131114851, -0.051706969967485715, 0.017654309421483834, -0.009402883971973454, 0.018665240454466964, 0.047317094908526196, -0.046461187408636265, 0.0700025515464211, 0.011592471859274084, 0.004511184467757278, 0.04387895065580499, -0.05758172035773275, -0.0652112796031097, 0.02938007657879512, 0.018718666130841836, -0.007416022181958594, 0.06681643154114748, -0.01198059249968629, -0.03786987565322164, 0.07552443510934619, 0.04143998471586489, 0.07684679544210404, 0.03012762850034315, 0.04271298496469336, 0.047821956932783155, -0.061811102240855564, -0.08536624821857247, 0.07272402477986013, 0.013850397673630464, -0.0032613208650047925, 0.09942246974508445, 0.03392199990450616, -0.004884070159740385, -0.01646420133036576, 0.062294052628459495, -0.05980846196531268, 0.02549782686655485, 0.019882767512619498, 0.0876651030885656, 0.04282713349556304, -0.028320751577595935, -0.08280658225910617, 0.052814729217465495, 0.017289476671680508, 0.02787013148590774, 0.006989142117697761, 0.022736082916060746, -5.9079852295061595e-05, -0.006636014093689104, -0.096786435751459, 0.03920742343732737, 0.002709411605701532, -0.012038600359330303, -0.019383945349776863, -0.01882759322904897, 0.0009775195487648992, 0.04779647400791964, -0.021924332260191165, -0.08014602967905132, 0.013594222460544328, 0.029164843014780388, -0.03611697378207659, -0.04322104343745194, 0.02666989677719563, 0.10824714004573864, 0.046019891665515844, 0.016434784415933806, -0.03233298289536236, 0.02613072938226784, -0.027856984693123508, 0.0637865559208752, 0.003961308077398015, -0.028357595968624565, -0.0444193687901826, -0.0512237568830843, 0.04526558804509597, 0.03841797075883823, 0.007314592005567633, 0.04967514173987281, -0.04898447784775915, 0.011150632812728, 0.007203831979738366, -0.061050012844988816, 0.06444306966845204, -0.038924812558364184, -0.009876526400245238, -0.019525214135071613, 0.05680598035149804, 0.02456710387949636, -0.0776012630879859, 0.011146063190601262, 0.05169249029159268, -0.04057475903439515, -0.032452211257444824, 0.013151979340367485, -0.028846249039843903, 0.03946767526622811, -0.04871223181360974, 0.023868959595513457, 0.02805825527944383, 0.03917556220021451, -0.0397176723787046, -0.025664968997205242, -0.0016095730856664691, 0.06983327565472262, 0.0007142184415019156, -0.05814004473432884, 0.08663291000774047, 0.022474551727156802, 0.08364215801744658, 0.05287400224720563, -0.001313064378853902, 0.02237557454511427, 0.04586192761520429, -0.04550799370625109, 0.008658539691101048, 0.03049895642448056, 0.0352995040705145, -0.01806194388308175, -0.04715756776660337, -0.0011595857431185303, 0.02646144084722541, 0.06194099223425971, -0.012169538922218942, -0.07108881589843374, 0.05193118883111744, -0.022682238185243394, -0.0010701276680705574, -0.0004567434618340778, 0.0023056634560874043, -0.0040145348375148875, 0.05736851006964163, 0.02141249931514771, 0.10149377307496403, -0.05825502700504697, 0.06915856375987606, 0.005192036273417149, 0.0808397879047688, -0.0066353728353467, -0.01619134529972043, -0.041514482504698684, -0.038340442560482066, 0.06198671221727685, 0.02003865720214962, -0.09774289646590481, 0.08021593655133011, -0.029919744487139206, -0.04813127912009629, 0.006281690097299726, 0.021192610045758318, 0.031735848394016206, 0.017631645153969408, -0.013395146312275638, -0.00558404523187022, -0.010306136413056393, -0.03442608177515237, -0.015365737550650242, -0.021957056808998465, 0.053841018102260524, 0.024748149697754962, -0.013506548247598199, 0.017548944129042027, -0.02402228251455592, -0.08163034018082176, -0.004863426048035253, 0.0037042678242451724, 0.030746428982439098, -0.04571275987090032, 0.009946788741838292, 0.004389313807912235, -0.018543085409065314, -0.034651975522700766, -0.022078459873936706, 0.04755760668583769, 0.04146157726707199, 0.06157279574946917, -0.01983501538565206, 0.05740205530952681, 0.02871715640209421, 0.04434414216697107, 0.004700473384353066, -0.028226683639473643, -0.09387346506157487, -0.11913656105982856, -0.058354661500517026, -0.020866522094165882, 0.11133354890920391, 0.05116623109855245, 0.027885008416287993, -0.010039603906076364, -0.004166233593758678, 0.020758583209631674, -0.035317861387050564, 0.028207430426229668, 0.01882650858777769, 0.012755917305748815, 0.035805576370409695, 0.03951044441585184, -0.026717902292079244, 0.050369212477159586, 0.01061639594433938, 0.01457020381861765, 0.024461381169679212, 0.0042220630117979765, -0.11195788217299392, 0.01313562826486592, 0.026794094621604987, -0.054996833378723985, -0.14378333931722723, -0.043671472667485854, 0.014081258949857924, -0.059950695740270364, -0.07000001711431748, -0.036664192249091766, 0.005951259131540929, -0.05376701810176264, -0.03364785870901252, -0.06823826718945279, 0.03372781448123548, -0.001409601789774687, 0.015297713389175061, 0.006357578247858289, 0.05244044744199295, -0.027779866935014535, -0.01627788815836229, 0.04605255061563418, 0.048331905563604276, 0.03823577307815014, 0.0012597472645891796, -0.03323293188081768, -0.029013761992024687, 0.003287898108598416, -0.0026429523308864563, -0.045867433618821805, -0.08848912950989243, 0.002020687365457438, -0.0021650205893654134, -0.0077275765491449325, 0.014621226691372273, 0.054846099726482585, -0.03724531844205642, -0.04934086654926184, 0.02572786642060368, -0.06833676691888041, -0.003308434337289748, 0.058532145930100095, -0.019816079021359685, 0.024606423226373113, -0.04203486072616521, -0.018176608092669477, 0.027883677497951776, 0.017357993793036826, 0.027204008820701675, -0.01045400921366253, -0.055621475026122996, -0.04311551974947667, 0.10252214343818102, -0.06976086618426751, 0.0232274708260234, 0.06736879859955876, 0.05171350151440454, -0.027145337194123882, 0.042781584961526585, -0.04445265178109796, 0.016879437625070538, 0.06236884211301332, -0.046881671083560734, 0.003964285692667104, 0.0885560486363967, -0.011623764498417852, -0.007337839810375545, 0.0025282692452461724, 0.013073450210154205, -0.02180940157289909, -0.009972805309762592, 0.012192749553850052, -0.054429157958312206, -0.046747340303253136, -0.013965852366627104, -0.0058033522983921065, 0.031379070554017445, 0.00330535557196476, -0.027059179829137466, -0.020618823753981247, 0.01985388303058293, -0.01343925254950076, 0.03239765171509341, -0.059744925823790054, 0.007657108878096325, -0.020593972609109137, -0.004122424649194935, 0.039183903815017546, 0.029494017928099244, -0.02836390692822825, 0.008628408655532535, 0.03212563514100505, 0.0042976982962193745, -0.011518701727568107, 0.057614412968650254, -0.03368051875961922, -0.009913462915877358, 0.04578426373947373, -0.09847198446696309, 0.016029009727057224, -0.005042112664786275, 0.048088652101978276, -0.004411061708998261, -0.009547113441868283]

# print >> sys.stderr, "value of A"
# print >> sys.stderr, str(A.tolist())
# print >> sys.stderr, "value of B"
# print >> sys.stderr, str(B.tolist())

net.make_input('inA', A)
net.make_input('inB', B)

net.make_array('A', N*subD, D/subD, dimensions=subD)
net.make_array('B', N*subD, D/subD, dimensions=subD)
net.make_array('D', N*subD, D/subD, dimensions=subD)

net.make_array('C', N_C, (D/2+1)*4, dimensions=2,
                   encoders=[[1,1],[1,-1],[-1,1],[-1,-1]], radius=3, num_subs=10)

AT = input_transform(D, True, False)
BT = input_transform(D, False, False)

net.connect('A', 'C', transform=AT, pstc=0.01)
net.connect('B', 'C', transform=BT, pstc=0.01)

ifftm2=output_transform(D)

net.connect('C', 'D', func=funcs.product, transform=ifftm2, pstc=0.01)

net.connect('inA', 'A')
net.connect('inB', 'B')

timesteps = 10000
dt_step = 0.01
pstc = 0.01

IAp = net.make_probe('inA', dt_sample=dt_step, pstc=pstc)
IBp = net.make_probe('inB', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
Cp = net.make_probe('C', dt_sample=dt_step, pstc=pstc)
Dp = net.make_probe('D', dt_sample=dt_step, pstc=pstc)

net.run(timesteps * dt_step)

iap_data = IAp.get_data()
ibp_data = IBp.get_data()
ap_data = Ap.get_data()
bp_data = Bp.get_data()
cp_data = Cp.get_data()
dp_data = Dp.get_data()

print "input 'inA' probe data"
for x in iap_data:
    print x
print "input 'inB' probe data"
for x in ibp_data:
    print x
print "ensemble 'A' probe data"
for x in ap_data:
    print x
print "ensemble 'B' probe data"
for x in bp_data:
    print x
print "ensemble 'C' probe data"
for x in cp_data:
    print x
print "ensemble 'D' probe data"
for x in dp_data:
    print x
