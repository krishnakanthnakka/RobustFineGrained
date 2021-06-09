
ATTACKS='fgsm1_2 fgsm1_8 bim10_2 bim10_8 pgd10_2 pgd10_8 mim10_2 mim10_8'

# To get results for AP*

for ATTACK in $ATTACKS
do
    # AP*
    #python run_adv_attack_AP.py -dataset=cub200  -config=settings_robust.yaml -mode=robust -split=test -backbone=vgg16 -net=AP -checkpoint=model.pth -attack=$ATTACK

    # ProtoPNet*
    python run_adv_attack_Proto.py  -dataset=cub200  -config=settings_robust.yaml -mode=robust -split=test -backbone=vgg16 -net=Proto -checkpoint=model.pth -attack=fgsm1_2


    # Ours-FR*
    python run_adv_attack_AttProto.py -dataset=cub200  -config=settings_robust.yaml -mode=robust -split=test -backbone=vgg16 -net=AttProto -checkpoint=model.pth -attack=$ATTACK -branch=FR

    #Ours-A*
    python run_adv_attack_AttProto.py -dataset=cub200  -config=settings_robust.yaml -mode=robust -split=test -backbone=vgg16 -net=AttProto -checkpoint=model.pth -attack=$ATTACK -branch=A


done
